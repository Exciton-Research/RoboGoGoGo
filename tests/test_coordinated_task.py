import base64
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from robogogogo.datasets import iter_transitions
from fixtures.api import create_app


def frame():
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 320, 240, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\x80" * 320 * 3) * 240)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def test_dual_arm_task_completes_and_exports_ordered_resource_ownership(tmp_path):
    with TestClient(create_app(project_dir=tmp_path, initial_scene="dual_arm")) as client:
        response = client.post("/api/tasks/plan", json={})
        assert response.status_code == 200, response.text
        snapshot = response.json()
        assert snapshot["coordination"]["stage_count"] == 5
        workspace = client.app.state.studio
        overlapping, errors = workspace.session.solve_targets(
            {arm.name: [0.55, 0, 0.75] for arm in workspace.session.arms}
        )
        assert np.max(np.asarray(errors)) < 0.005
        assert workspace.task_service.world.clearance(overlapping).distance < 0

        assert client.post("/api/command", json={"op": "run"}).status_code == 422
        assert client.post("/api/tasks/step", json={"observation": 0}).status_code == 422
        paused = client.post("/api/tasks/pause", json={}).json()
        assert paused["coordination"]["status"] == "paused"
        assert client.post("/api/tasks/step", json={"observation": 0}).status_code == 422
        snapshot = client.post("/api/tasks/resume", json={}).json()
        image = frame()

        def capture(snapshot):
            response = client.post(
                "/api/episodes/observe",
                json={
                    "images": {c["name"]: image for c in snapshot["episode"]["cameras"]},
                    "sequence": snapshot["sequence"],
                },
            )
            assert response.status_code == 200, response.text
            return response.json()

        for _ in range(180):
            observation = capture(snapshot)
            response = client.post(
                "/api/tasks/step", json={"observation": observation["observation"]}
            )
            assert response.status_code == 200, response.text
            snapshot = response.json()
            assert snapshot["coordination"]["minimum_clearance_m"] >= 0.015
            if snapshot["coordination"]["status"] in ("completed", "blocked"):
                break
        assert snapshot["coordination"]["status"] == "completed", snapshot["coordination"]
        assert snapshot["episode"]["success"]
        events = snapshot["coordination"]["events"]
        assert [e["owner"] for e in events] == [None, "left_arm", None, "right_arm", None]
        assert all(a["completed_at"] < b["completed_at"] for a, b in zip(events, events[1:]))
        capture(snapshot)
        result = client.post("/api/episodes/save", json={})
        assert result.status_code == 200, result.text
        path = tmp_path / "episodes" / f"{result.json()['id']}.zip"
        transitions = list(iter_transitions(path))
        assert transitions and all(t.source == "task_planner" for t in transitions)
        assert transitions[-1].terminated and not transitions[-1].truncated
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("episode.json"))
            assert len(manifest["task_plan"]["stages"]) == 5
            assert manifest["task_progress"]["status"] == "completed"
            assert all(t["coordination"]["clearance_m"] >= 0.015 for t in manifest["transitions"])
        # A new standalone episode clears planner ownership and obstacle state.
        reset = client.post("/api/episodes/reset", json={}).json()
        assert reset["coordination"] is None


def test_invalid_plan_preserves_existing_episode(tmp_path):
    with TestClient(create_app(project_dir=tmp_path, initial_scene="dual_arm")) as client:
        before = client.get("/api/snapshot").json()
        response = client.post("/api/tasks/plan", json={"station": [50, 0, 50]})
        assert response.status_code == 422
        after = client.get("/api/snapshot").json()
        assert after["episode"]["id"] == before["episode"]["id"]
        np.testing.assert_array_equal(after["state"]["q"], before["state"]["q"])


def test_runtime_guard_rejects_zero_time_action_without_corrupting_dataset(tmp_path):
    import jax.numpy as jnp

    from robogogogo.spatial import State

    with TestClient(create_app(project_dir=tmp_path, initial_scene="dual_arm")) as client:
        client.post("/api/tasks/plan", json={}).raise_for_status()
        workspace = client.app.state.studio
        with workspace.lock:
            session = workspace.session
            before = np.asarray(session.state.q).copy()
            session.state = State(session.state.q, jnp.ones_like(session.state.q) * 1000)
            workspace._refresh()
        snapshot = client.get("/api/snapshot").json()
        captured = client.post(
            "/api/episodes/observe",
            json={
                "images": {c["name"]: frame() for c in snapshot["episode"]["cameras"]},
                "sequence": snapshot["sequence"],
            },
        ).json()
        result = client.post("/api/tasks/step", json={"observation": captured["observation"]})
        assert result.status_code == 200, result.text
        after = result.json()
        assert after["coordination"]["status"] == "blocked"
        assert after["episode"]["actions"] == 0
        assert after["state"]["time"] == snapshot["state"]["time"]
        np.testing.assert_array_equal(after["state"]["q"], before)
        saved = client.post("/api/episodes/save", json={})
        assert saved.status_code == 200, saved.text
        path = tmp_path / "episodes" / f"{saved.json()['id']}.zip"
        assert list(iter_transitions(path)) == []


def test_handover_picks_transfers_places_and_records_payload(tmp_path):
    from robogogogo.studio.schemas import CameraObservation, PlanTask
    from fixtures.workspace import Studio

    workspace = Studio(tmp_path, scene="dual_arm")
    snapshot = workspace.task_service.plan(PlanTask(kind="handover"))
    image = frame()
    owners = []
    for _ in range(450):
        captured = workspace.episode_service.capture_observation(
            CameraObservation(
                images={c["name"]: image for c in workspace.episode.cameras},
                sequence=workspace.sequence,
            )
        )
        snapshot = workspace.task_service.step(captured["observation"])
        owner = snapshot["manipulation"]["object"]["owner"]
        if not owners or owner != owners[-1]:
            owners.append(owner)
        assert snapshot["coordination"]["minimum_clearance_m"] >= 0.015
        if snapshot["coordination"]["status"] in ("completed", "blocked"):
            break
    assert snapshot["coordination"]["status"] == "completed", snapshot["state"]["error"]
    assert owners == [None, "left_arm", "right_arm", None]
    assert snapshot["episode"]["success"]
    assert workspace.session.payload_model is None
    assert snapshot["manipulation"]["grasp_model"] == "compliant_frictional_contact"
    assert any(
        all(g["holding"] for g in o["manipulation"]["grippers"].values())
        for o in workspace.episode.observations
    )
    assert any(
        np.linalg.norm(o["manipulation"]["reaction_torque"]) > 0.1
        for o in workspace.episode.observations
    )
    np.testing.assert_allclose(
        snapshot["manipulation"]["object"]["position"],
        workspace.task_service.handover_scene.place,
        atol=0.015,
    )
    assert [
        e["event"]
        for e in snapshot["manipulation"]["events"]
        if e["event"] in {"grasp", "receiver_secured", "handover", "release", "placement_verified"}
    ] == [
        "grasp",
        "receiver_secured",
        "handover",
        "release",
        "placement_verified",
    ]
    assert all(not g["holding"] for g in snapshot["manipulation"]["grippers"].values())
    assert any(
        o["manipulation"]["object"]["owner"] == "right_arm" for o in workspace.episode.observations
    )
    workspace.episode_service.capture_observation(
        CameraObservation(
            images={c["name"]: image for c in workspace.episode.cameras},
            sequence=workspace.sequence,
        )
    )
    saved = workspace.episode_service.save_episode()
    path = tmp_path / "episodes" / f"{saved['id']}.zip"
    transitions = list(iter_transitions(path))
    assert transitions[-1].terminated and not transitions[-1].truncated
    assert any(t.observation.manipulation["object"]["owner"] == "right_arm" for t in transitions)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("episode.json"))
        assert manifest["task_plan"]["payload"]["mass"] == 0.25
        assert any(a.get("grasp_events") for a in manifest["transitions"])


def test_material_grasp_fixture_lifts_soft_object_and_exports_damage_state(tmp_path):
    from robogogogo.studio.schemas import CameraObservation, PlanTask
    from fixtures.workspace import Studio

    workspace = Studio(tmp_path, scene="dual_arm")
    snapshot = workspace.task_service.plan(
        PlanTask(
            kind="material_grasp",
            object_material="soft_foam",
            initial_grip_force_n=4,
            object_damage_force_n=10,
        )
    )
    image = frame()
    for _ in range(300):
        captured = workspace.episode_service.capture_observation(
            CameraObservation(
                images={c["name"]: image for c in workspace.episode.cameras},
                sequence=workspace.sequence,
            )
        )
        snapshot = workspace.task_service.step(captured["observation"])
        if snapshot["coordination"]["status"] in ("completed", "blocked"):
            break
    assert snapshot["coordination"]["status"] == "completed", snapshot["state"]["error"]
    assert snapshot["episode"]["success"]
    material = snapshot["manipulation"]["object"]["material"]
    assert not material["damaged"]
    assert material["maximum_indentation_m"] > 0.0005
    assert snapshot["manipulation"]["object"]["position"][2] > 0.54
    assert any(e["event"] == "material_lift_verified" for e in snapshot["manipulation"]["events"])
    assert workspace.episode.task_plan["object_material"]["parameters"]["modulus_pa"] == 180_000
    assert workspace.episode.observations[-1]["manipulation"]["object"]["material"]["indentation_m"]
