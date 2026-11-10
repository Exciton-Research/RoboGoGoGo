"""Studio cannot route command tests into read-only hardware or enable real motion."""

from fastapi.testclient import TestClient

from robogogogo.studio.api import create_app


def test_execution_is_disabled_by_default(tmp_path):
    with TestClient(create_app(project_dir=tmp_path, initial_scene="drop")) as client:
        state = client.get("/api/execution").json()
        assert state["mode"] == "disabled" and not state["physical_execution_enabled"]
        response = client.post("/api/execution/arm", json={"owner": "manual", "expected_epoch": 0})
        assert response.status_code == 503
        assert client.post("/api/execution/stop", json={}).status_code == 503


def test_local_api_coordinates_takeover_stop_and_revokes_old_commands(tmp_path):
    app = create_app(project_dir=tmp_path, initial_scene="drop", test_execution=True)
    with TestClient(app) as client:
        executor = app.state.execution
        state = client.get("/api/execution").json()
        assert state["backend"] == "test" and state["stop_confirmed"]
        lease = client.post(
            "/api/execution/arm", json={"owner": "policy", "expected_epoch": state["epoch"]}
        ).json()
        state = client.get("/api/execution").json()
        body = dict(
            token=lease["token"],
            sequence=1,
            observation_sequence=state["feedback"]["sequence"],
            left=[0.2] * 6,
            right=[-0.2] * 6,
            left_gripper={"width": 0.02, "force_limit": 5.0},
            right_gripper={"width": 0.04, "force_limit": 5.0},
            duration_s=2.0,
        )
        result = client.post("/api/execution/command", json=body)
        result.raise_for_status()
        assert result.json()["snapshot"]["mode"] == "executing"
        assert not result.json()["snapshot"]["physical_execution_enabled"]
        assert lease["token"] not in client.get("/api/execution").text
        invalid = dict(body, unknown=True)
        assert client.post("/api/execution/command", json=invalid).status_code == 422
        manual = client.post("/api/execution/takeover", json={"expected_epoch": lease["epoch"]})
        manual.raise_for_status()
        assert manual.json()["snapshot"]["owner"] == "manual"
        assert client.post("/api/execution/command", json=body).status_code == 422
        assert (
            client.post("/api/execution/heartbeat", json={"token": lease["token"]}).status_code
            == 422
        )
        current = manual.json()
        assert (
            client.post("/api/execution/heartbeat", json={"token": current["token"]}).status_code
            == 200
        )
        stop = client.post("/api/execution/stop", json={}).json()
        assert stop["mode"] == "stopped" and stop["stop_confirmed"]
        assert (
            client.post(
                "/api/execution/arm", json={"owner": "policy", "expected_epoch": current["epoch"]}
            ).status_code
            == 422
        )
        assert (
            client.post("/api/execution/heartbeat", json={"token": current["token"]}).status_code
            == 422
        )
        workflow = client.get("/api/workflow").json()
        assert workflow["command_test"]["backend"] == "test"
        assert not workflow["physical_execution_enabled"]
        assert client.get("/api/hardware").json()["backend"] is None
        assert client.post("/api/execution/reset", json={}).status_code == 200
        assert (
            client.post(
                "/api/execution/stop", json={}, headers={"Origin": "https://other.example"}
            ).status_code
            == 403
        )
    assert executor.backend.closed and executor.mode == "closed"
    assert executor.thread is not None and not executor.thread.is_alive()
