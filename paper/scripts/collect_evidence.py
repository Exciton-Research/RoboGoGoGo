"""Reproduce the report's bounded software checks; requires robogogogo's dev extras."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from datetime import UTC, datetime
from pathlib import Path

SUITES = (
    "tests/test_robot_product.py",
    "tests/test_workflow_layers.py",
    "tests/test_execution.py",
)


def png(width, height, value):
    """A synthetic solid RGB frame; exercises serialization, not visual rendering."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    row = b"\0" + bytes([value, value, value]) * width
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report_dir = Path(__file__).resolve().parents[1]
    evidence = report_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    source = repo / "src"
    sys.path.insert(0, str(source))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.update(
        {
            k: env[k]
            for k in ("JAX_PLATFORMS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    )
    if dirty:
        raise SystemExit(
            "Use a clean source checkout so the recorded commit identifies the code."
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "numpy", "pytest", "fastapi", "httpx")
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-o",
        "faulthandler_timeout=90",
        *SUITES,
        "--junitxml=" + str(evidence / "regressions.xml"),
    ]
    print("Running bounded regressions", flush=True)
    with (evidence / "regressions.txt").open("w") as log:
        result = subprocess.run(
            command,
            check=False,
            cwd=repo,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    print((evidence / "regressions.txt").read_text(), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    suites = ET.parse(evidence / "regressions.xml").getroot()
    cases = list(suites.iter("testcase"))
    regressions = {
        "tests": len(cases),
        "failures": len(list(suites.iter("failure"))),
        "errors": len(list(suites.iter("error"))),
        "skipped": len(list(suites.iter("skipped"))),
        "files": list(SUITES),
        "cases": [
            {"file": c.attrib.get("classname"), "name": c.attrib["name"]} for c in cases
        ],
    }
    import jax

    jax.config.update("jax_enable_x64", True)
    from tempfile import TemporaryDirectory

    from fastapi.testclient import TestClient
    from robogogogo.robot.definition import ROBOT_ID, ROBOT_REVISION
    from robogogogo.robot.validate_working_range import audit
    from robogogogo.studio.api import create_app

    print("Running seed-42 geometry audit", flush=True)
    geometry = audit(count=24, seed=42)
    (evidence / "practical-range.json").write_text(
        json.dumps(geometry, indent=2) + "\n"
    )
    assert geometry["status"] == "sampled_clear"
    print(
        "Running production API capture/save/validate check with synthetic frames",
        flush=True,
    )
    with TemporaryDirectory(prefix="robogogogo-report-") as workspace:
        with TestClient(create_app(project_dir=workspace)) as client:
            initial = client.get("/api/snapshot").json()
            cameras = initial["episode"]["cameras"]

            def capture():
                state = client.get("/api/snapshot").json()
                response = client.post(
                    "/api/episodes/observe",
                    json={
                        "sequence": state["sequence"],
                        "images": {
                            c["name"]: png(c["width"], c["height"], i * 50)
                            for i, c in enumerate(cameras)
                        },
                    },
                )
                response.raise_for_status()
                return response.json()["observation"]

            before = capture()
            response = client.post(
                "/api/episodes/act",
                json={
                    "delta": [0.0] * initial["model"]["dof"],
                    "observation": before,
                    "control_ticks": 1,
                    "source": "manual",
                },
            )
            response.raise_for_status()
            after = capture()
            saved_response = client.post("/api/episodes/save")
            saved_response.raise_for_status()
            saved = saved_response.json()
            response = client.post(f"/api/data/{saved['id']}/validate", json={})
            response.raise_for_status()
            validation = response.json()
            final = client.get("/api/snapshot").json()
            workflow = client.get("/api/workflow").json()
            assert validation["transitions"] == 1 and len(validation["cameras"]) == 4
            assert validation["environment"] == "simulation"
            assert not validation["physical_deployment_ready"]
            assert not workflow["physical_execution_enabled"]
            smoke = {
                "input_images": "Synthetic solid-color PNGs, not rendered robot observations",
                "action": "One zero-delta 31-coordinate manual action, control_ticks=1",
                "observation_indices": [before, after],
                "time_before_s": initial["state"]["time"],
                "time_after_s": final["state"]["time"],
                "validation": validation,
                "physical_execution_enabled": workflow["physical_execution_enabled"],
                "archive_retained": False,
            }
    historical = repo / "docs/validation/working-range-engineering-11.json"
    report = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "source_dirty": dirty,
        "robot": {"id": ROBOT_ID, "revision": ROBOT_REVISION},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "jax_backend": jax.default_backend(),
        "jax_x64": True,
        "regressions": regressions,
        "practical_geometry": {
            k: geometry[k] for k in ("seed", "counts", "margin_m", "status")
        },
        "capture_smoke": smoke,
        "historical_audit_sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
        "limits": [
            "Targeted software regressions only; not the full project suite",
            "Synthetic capture frames do not test WebGL appearance",
            "No policy training, physical hardware, or cross-simulator benchmark",
        ],
    }
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "commit": commit,
                "regressions": len(cases),
                "geometry": geometry["counts"],
                "transitions": validation["transitions"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
