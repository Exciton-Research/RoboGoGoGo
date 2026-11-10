"""Single-robot workflow boundaries and the capture-to-training-input path."""

import importlib
import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from fastapi.testclient import TestClient
from fixtures.api import create_app
from test_vla import png_observation

from robogogogo.robot.profile import RobotProfile
from robogogogo.runtime.workflow import RobotWorkflow


def test_legacy_imports_preserve_contract_identity():
    for old, new in (
        ("hardware.contracts", "robot.contracts"),
        ("hardware.calibration", "robot.calibration"),
        ("vla.contracts", "runtime.contracts"),
        ("vla.recording", "data.recording"),
        ("datasets", "data.datasets"),
        ("system", "runtime.supervisor"),
        ("policies", "runtime.policies"),
    ):
        assert importlib.import_module(f"robogogogo.{old}") is importlib.import_module(
            f"robogogogo.{new}"
        )


def test_robot_and_dataset_layers_do_not_initialize_simulation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import robogogogo.robot.profile; "
            "import robogogogo.data.datasets; import robogogogo.learning.datasets; "
            "assert 'jax' not in sys.modules; assert 'fastapi' not in sys.modules",
        ],
        check=True,
        timeout=15,
    )


def test_profile_requires_one_distinct_dual_arm_mapping(tmp_path):
    profile = RobotProfile()
    assert "model" in profile.missing()
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(asdict(profile)))
    assert RobotProfile.load(path).digest == profile.digest
    with pytest.raises(ValueError, match="distinct joints"):
        replace(profile, left_joints=("shared",), right_joints=("shared",))
    with pytest.raises(ValueError, match="four distinct"):
        replace(profile, cameras=("head",))
    with pytest.raises(ValueError, match="own gripper"):
        replace(profile, left_gripper="gripper", right_gripper="gripper")
    with pytest.raises(ValueError, match="seven mapped"):
        replace(profile, left_joints=("l",), right_joints=("r",))
    data = asdict(profile)
    data["third_arm"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly"):
        RobotProfile.load(path)


def test_read_only_calibrated_telemetry_cannot_enable_physical_execution(tmp_path):
    robot = RobotProfile(
        model="test_model",
        backend="robot",
        left_joints=tuple(f"l{i}" for i in range(7)),
        right_joints=tuple(f"r{i}" for i in range(7)),
        left_gripper="lg",
        right_gripper="rg",
        cameras=("hl", "hr", "wl", "wr"),
    )
    workflow = RobotWorkflow(tmp_path, robot=robot)
    hardware = dict(
        backend="robot",
        status="streaming",
        joints={"names": list(robot.left_joints + robot.right_joints)},
        cameras=[{"name": n} for n in robot.cameras],
        perception={"state": "observing"},
    )
    result = workflow.snapshot(hardware)
    assert result["hardware"]["calibrated"]
    assert not result["physical_execution_enabled"]
    assert next(s for s in result["stages"] if s["id"] == "runtime")["state"] == "blocked"
    hardware["joints"]["names"] = ["wrong"]
    assert not workflow.snapshot(hardware)["hardware"]["calibrated"]
    hardware["backend"] = "other_robot"
    assert not workflow.snapshot(hardware)["hardware"]["connected"]


def test_dual_arm_capture_save_validate_and_reload_workflow(tmp_path):
    with TestClient(create_app(project_dir=tmp_path, initial_scene="mobile_dual_arm")) as client:
        snapshot = client.get("/api/snapshot").json()
        assert snapshot["model"]["scene"] == "mobile_dual_arm"
        assert len(snapshot["model"]["arms"]) == 2
        workflow = client.get("/api/workflow").json()
        assert workflow["robot"]["model"] is None
        assert not workflow["physical_execution_enabled"]
        cameras = snapshot["episode"]["cameras"]
        assert len(cameras) == 4

        def capture():
            state = client.get("/api/snapshot").json()
            response = client.post(
                "/api/episodes/observe",
                json={
                    "sequence": state["sequence"],
                    "images": {c["name"]: png_observation() for c in cameras},
                },
            )
            response.raise_for_status()
            return response.json()["observation"]

        index = capture()
        client.post(
            "/api/episodes/act",
            json={
                "delta": [0.0] * snapshot["model"]["dof"],
                "observation": index,
                "control_ticks": 1,
                "source": "manual",
            },
        ).raise_for_status()
        capture()
        saved = client.post("/api/episodes/save").json()
        assert saved["environment"] == "simulation"
        report = client.post(f"/api/data/{saved['id']}/validate", json={})
        report.raise_for_status()
        assert report.json()["transitions"] == 1
        assert len(report.json()["cameras"]) == 4
        assert report.json()["environment"] == "simulation"
        assert not report.json()["physical_deployment_ready"]
        assert client.post("/api/data/not-an-id/validate", json={}).status_code == 422
        assert (
            client.post(
                "/api/data/00000000-0000-0000-0000-000000000000/validate", json={}
            ).status_code
            == 404
        )
    workflow = RobotWorkflow(tmp_path)
    assert workflow.episodes.list()[0]["id"] == saved["id"]
    assert workflow.validate_episode(saved["id"])["sha256"] == report.json()["sha256"]


def test_malformed_dataset_is_reported_as_validation_error(tmp_path):
    workflow = RobotWorkflow(tmp_path)
    workflow.episodes.directory.mkdir()
    identifier = "00000000-0000-0000-0000-000000000001"
    workflow.episodes.archive(identifier).write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="malformed"):
        workflow.validate_episode(identifier)
