"""API integration tests against the native JAX engine, not mocked simulation."""

import io
import json
import time
from importlib.resources import files
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from fixtures.api import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    directory = tmp_path_factory.mktemp("studio-projects")
    with TestClient(create_app(project_dir=directory, initial_scene="drop")) as client:
        yield client


@pytest.fixture(autouse=True)
def reset_scene(client):
    client.post("/api/command", json={"op": "scene", "scene": "drop"}).raise_for_status()


def test_snapshot_and_single_step_use_real_physics(client):
    before = client.get("/api/snapshot").json()
    after = client.post("/api/command", json={"op": "step"}).json()
    assert before["model"]["dof"] == 1
    assert not before["state"]["running"]
    assert after["state"]["time"] > before["state"]["time"]
    assert after["state"]["q"][0] < before["state"]["q"][0]
    assert after["state"]["samples"] == 2
    assert after["sequence"] > before["sequence"]


def test_goals_do_not_teleport_robot_and_reset_clears_history(client):
    before = client.get("/api/snapshot").json()
    goal = client.post("/api/command", json={"op": "joints", "values": [0.8]}).json()
    assert goal["state"]["q"] == before["state"]["q"]
    np.testing.assert_allclose(goal["state"]["goal"], [0.8])
    reset = client.post("/api/command", json={"op": "reset"}).json()
    assert reset["state"]["samples"] == 1
    assert reset["epoch"] > before["epoch"]


def test_project_survives_disk_roundtrip_and_restarts_recording(client):
    client.post("/api/command", json={"op": "step"}).raise_for_status()
    saved = client.post("/api/projects", json={"name": "Contact experiment"}).json()
    project = saved["project"]
    assert project["format"] == "robogogogo-project-v1"
    identifier = project["id"]
    listing = client.get("/api/projects").json()
    assert any(p["id"] == identifier for p in listing)
    client.post("/api/command", json={"op": "step"})
    restored = client.post(f"/api/projects/{identifier}/load").json()
    np.testing.assert_array_equal(restored["state"]["q"], project["q"])
    assert restored["state"]["time"] == project["elapsed"]
    assert restored["state"]["samples"] == 1
    assert not restored["state"]["running"]
    updated = client.post(
        "/api/projects", json={"name": "Renamed experiment", "id": identifier}
    ).json()
    assert updated["project"]["id"] == identifier
    assert updated["snapshot"]["project"]["name"] == "Renamed experiment"
    assert (client.app.state.studio.directory / f"{identifier}.json").exists()


def test_trajectory_download_is_complete_npz(client):
    for _ in range(3):
        client.post("/api/command", json={"op": "step"}).raise_for_status()
    response = client.get("/api/export")
    response.raise_for_status()
    assert "robogogogo-" in response.headers["content-disposition"]
    with np.load(io.BytesIO(response.content), allow_pickle=False) as recording:
        assert recording["q"].shape == (4, 1)
        assert json.loads(str(recording["metadata"]))["format"] == "robogogogo-trajectory-v1"
        assert "model_inertia" in recording.files


@pytest.mark.parametrize(
    "body",
    [
        {"op": "joints", "values": [0, 1]},
        {"op": "joints"},
        {"op": "target", "values": [0, 0, 0]},
        {"op": "scene"},
        {"op": "speed", "speed": 100},
        {"op": "destroy"},
        {"op": "run", "extra": 1},
    ],
)
def test_invalid_commands_are_actionable_422_errors(client, body):
    response = client.post("/api/command", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_no_external_origins_or_arbitrary_project_paths(client):
    response = client.post(
        "/api/command", json={"op": "run"}, headers={"Origin": "https://example.com"}
    )
    assert response.status_code == 403
    assert client.post("/api/projects", json={"name": "bad", "id": "../outside"}).status_code == 422
    assert client.post(f"/api/projects/{uuid4()}/load").status_code == 404
    assert client.post("/api/projects", json={"name": "   "}).status_code == 422


def test_bad_project_load_is_transactional(client):
    saved = client.post("/api/projects", json={"name": "Broken model test"}).json()["project"]
    before = client.get("/api/snapshot").json()
    path = client.app.state.studio.directory / f"{saved['id']}.json"
    saved.update(q=[0, 1], qd=[0, 0], goal=[0, 0])
    path.write_text(json.dumps(saved))
    assert client.post(f"/api/projects/{saved['id']}/load").status_code == 422
    after = client.get("/api/snapshot").json()
    assert after["epoch"] == before["epoch"]
    assert after["state"]["q"] == before["state"]["q"]


def test_websocket_stream_clock_and_last_disconnect_pause(client):
    with client.websocket_connect("/api/ws") as websocket:
        initial = websocket.receive_json()
        client.post("/api/command", json={"op": "run"}).raise_for_status()
        for _ in range(30):
            sample = websocket.receive_json()
            if sample["state"]["time"] > initial["state"]["time"]:
                break
        assert sample["state"]["running"]
        assert sample["state"]["time"] > initial["state"]["time"]
    for _ in range(30):
        if not client.get("/api/snapshot").json()["state"]["running"]:
            break
        time.sleep(0.01)
    assert not client.get("/api/snapshot").json()["state"]["running"]


def test_imported_robot_is_real_and_bad_import_preserves_workspace(client):
    xml = files("robogogogo").joinpath("assets/mobile_arm_chain.urdf").read_text()
    response = client.post("/api/robot", json={"xml": xml})
    response.raise_for_status()
    model = response.json()["model"]
    assert model["scene"] == "urdf" and model["dof"] == 6
    assert len(model["visuals"]) == 8
    before = client.get("/api/snapshot").json()
    assert client.post("/api/robot", json={"xml": '<robot name="empty"/>'}).status_code == 422
    assert client.post("/api/robot", json={"xml": "<broken"}).status_code == 422
    assert client.get("/api/snapshot").json()["epoch"] == before["epoch"]


def test_recording_cap_stops_physics(client):
    studio = client.app.state.studio
    original = studio.max_samples
    studio.max_samples = 3
    try:
        client.post("/api/command", json={"op": "run"})
        studio.tick()
        studio.tick()
        snapshot = client.get("/api/snapshot").json()
        assert not snapshot["state"]["running"]
        assert snapshot["state"]["samples"] == 3
        assert client.post("/api/command", json={"op": "run"}).status_code == 422
    finally:
        studio.max_samples = original
