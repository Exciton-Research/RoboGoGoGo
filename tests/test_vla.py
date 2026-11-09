"""Aligned VLA data contract: RGB_t, instruction, state_t, action_t, state_t+1."""

import base64
import io
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from robogogogo.server import create_app
from robogogogo.vla import decode_camera_png


def png_observation():
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    header = struct.pack(">IIBBBBB", 320, 240, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + b"\x80\xa0\xc0" * 320) * 240
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(project_dir=tmp_path_factory.mktemp("vla"))) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_episode(client):
    client.post(
        "/api/vla/reset", json={"instruction": "Reach the highlighted target.", "variant": "a"}
    ).raise_for_status()


def capture(client):
    snapshot = client.get("/api/snapshot").json()
    response = client.post(
        "/api/vla/observe", json={"image": png_observation(), "sequence": snapshot["sequence"]}
    )
    response.raise_for_status()
    return response.json()


def test_reset_has_explicit_model_agnostic_contract(client):
    snapshot = client.get("/api/snapshot").json()
    episode = snapshot["episode"]
    assert episode["instruction"] == "Reach the highlighted target."
    assert episode["action_spec"]["representation"] == "joint_position_delta"
    assert episode["action_spec"]["dimensions"] == 6
    assert episode["action_spec"]["normalization"] == "none"
    assert episode["action_spec"]["gripper"] is False
    assert snapshot["state"]["distance"] > 0.02
    assert not snapshot["state"]["running"]


def test_capture_action_and_next_observation_are_aligned(client):
    before = capture(client)
    result = client.post(
        "/api/vla/act",
        json={
            "delta": [0.3, 0, 0, 0, 0, 0],
            "observation": before["observation"],
            "control_ticks": 10,
            "source": "manual",
        },
    )
    result.raise_for_status()
    after = result.json()
    assert after["episode"]["actions"] == 1
    np.testing.assert_allclose(after["state"]["target"], before["snapshot"]["state"]["target"])
    assert after["state"]["time"] - before["snapshot"]["state"]["time"] == pytest.approx(0.1)
    assert after["state"]["q"] != before["snapshot"]["state"]["q"]
    next_observation = capture(client)
    assert next_observation["observation"] == 1
    saved = client.post("/api/vla/save").json()
    archive_response = client.get(f"/api/vla/episodes/{saved['id']}")
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        manifest = json.loads(archive.read("episode.json"))
        assert manifest["format"] == "robogogogo-episode-v1"
        transition = manifest["transitions"][0]
        assert transition["observation"] == 0 and transition["next_observation"] == 1
        assert transition["applied_delta"][0] == pytest.approx(0.12, abs=1e-6)
        assert manifest["observations"][1]["time"] == transition["next_time"]
        assert manifest["missing_next_observations"] == 0
        assert archive.read("frames/000000.png").startswith(b"\x89PNG")
        assert "physics.npz" in archive.namelist()
    assert any(e["id"] == saved["id"] for e in client.get("/api/vla/episodes").json())


def test_stale_or_missing_observations_are_rejected(client):
    assert client.post("/api/vla/act", json={"delta": [0] * 6, "observation": 0}).status_code == 422
    stale = client.get("/api/snapshot").json()
    observed = capture(client)
    assert (
        client.post(
            "/api/vla/observe", json={"image": png_observation(), "sequence": stale["sequence"]}
        ).status_code
        == 422
    )
    client.post(
        "/api/vla/act", json={"delta": [0.1] * 6, "observation": observed["observation"]}
    ).raise_for_status()
    assert (
        client.post(
            "/api/vla/act", json={"delta": [0] * 6, "observation": observed["observation"]}
        ).status_code
        == 422
    )


def test_debug_controls_invalidate_episode_instead_of_mislabeling_data(client):
    capture(client)
    result = client.post("/api/command", json={"op": "step"}).json()
    assert result["episode"]["done"]
    assert "outside" in result["episode"]["reason"]
    assert client.post("/api/vla/act", json={"delta": [0] * 6, "observation": 0}).status_code == 422


def test_reference_policy_is_labeled_and_preserves_observation_action_order(client):
    observed = capture(client)
    for _ in range(65):
        reference = client.get("/api/vla/reference-action").json()
        assert reference["source"] == "scripted_reference"
        assert len(reference["delta"]) == 6
        result = client.post(
            "/api/vla/act",
            json=dict(
                delta=reference["delta"],
                observation=observed["observation"],
                source="scripted_reference",
            ),
        ).json()
        observed = capture(client)
        if result["episode"]["done"]:
            break
    assert result["episode"]["success"]
    assert result["state"]["distance"] < 0.02
    assert (
        result["episode"]["observations"] == result["episode"]["actions"]
    )  # Before the final capture.
    assert (
        observed["snapshot"]["episode"]["observations"]
        == observed["snapshot"]["episode"]["actions"] + 1
    )


def test_invalid_images_and_actions_are_rejected(client):
    with pytest.raises(ValueError):
        decode_camera_png("data:image/png;base64,abc")
    response = client.post("/api/vla/reset", json={"instruction": "  "})
    assert response.status_code == 422
    observed = capture(client)
    assert (
        client.post(
            "/api/vla/act", json={"delta": [0, 0], "observation": observed["observation"]}
        ).status_code
        == 422
    )


def test_dataset_reader_preserves_images_and_action_source(client, tmp_path):
    from robogogogo.datasets import iter_transitions

    observed = capture(client)
    client.post(
        "/api/vla/act",
        json={
            "delta": [0.05, 0, 0, 0, 0, 0],
            "observation": observed["observation"],
            "source": "manual",
        },
    ).raise_for_status()
    capture(client)
    saved = client.post("/api/vla/save").json()
    path = tmp_path / "episode.zip"
    path.write_bytes(client.get(f"/api/vla/episodes/{saved['id']}").content)
    transitions = list(iter_transitions(path))
    assert len(transitions) == 1
    assert transitions[0].source == "manual"
    assert transitions[0].observation.instruction == "Reach the highlighted target."
    assert transitions[0].observation.image_png.startswith(b"\x89PNG")
    assert transitions[0].next_observation.time > transitions[0].observation.time


def test_reader_rejects_missing_next_camera_frame(client, tmp_path):
    from robogogogo.datasets import iter_transitions

    observed = capture(client)
    client.post(
        "/api/vla/act",
        json={"delta": [0.05, 0, 0, 0, 0, 0], "observation": observed["observation"]},
    ).raise_for_status()
    saved = client.post("/api/vla/save").json()
    path = tmp_path / "incomplete.zip"
    path.write_bytes(client.get(f"/api/vla/episodes/{saved['id']}").content)
    with pytest.raises(ValueError, match="aligned camera"):
        list(iter_transitions(path))


def test_multicamera_bundle_is_atomic_and_round_trips(client, tmp_path):
    cameras = [
        dict(name="head_left", mount="pedestal", position=[0, -0.04, 1.2], look_at=[1, -0.04, 0.4]),
        dict(name="head_right", mount="pedestal", position=[0, 0.04, 1.2], look_at=[1, 0.04, 0.4]),
        dict(name="wrist", mount="tool", position=[0, 0, 0.1], look_at=[1, 0, 0.1]),
    ]
    response = client.post("/api/vla/reset", json=dict(cameras=cameras))
    response.raise_for_status()
    snapshot = response.json()
    assert len(snapshot["episode"]["cameras"]) == 3
    assert snapshot["episode"]["cameras"][2]["body"] == snapshot["model"]["names"].index("tool")
    assert snapshot["episode"]["cameras"][0]["intrinsics"][0][0] > 0
    image = png_observation()
    incomplete = client.post(
        "/api/vla/observe", json=dict(sequence=snapshot["sequence"], images={"head_left": image})
    )
    assert incomplete.status_code == 422
    assert client.get("/api/snapshot").json()["episode"]["observations"] == 0
    images = {camera["name"]: image for camera in cameras}
    result = client.post(
        "/api/vla/observe", json=dict(sequence=snapshot["sequence"], images=images)
    )
    result.raise_for_status()
    snapshot = client.post(
        "/api/vla/act", json=dict(delta=[0.05, 0, 0, 0, 0, 0], observation=0)
    ).json()
    result = client.post(
        "/api/vla/observe", json=dict(sequence=snapshot["sequence"], images=images)
    )
    result.raise_for_status()
    saved = client.post("/api/vla/save", json={}).json()
    archive_path = tmp_path / "multiview.zip"
    archive_path.write_bytes(client.get(f"/api/vla/episodes/{saved['id']}").content)
    from robogogogo.datasets import iter_transitions

    transition = next(iter_transitions(archive_path))
    assert set(transition.observation.images) == set(images)
    assert len(transition.observation.cameras) == 3
    assert set(transition.observation.camera_poses) == set(images)
    assert (
        transition.observation.camera_poses["wrist"]
        != transition.next_observation.camera_poses["wrist"]
    )
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("episode.json"))
        assert manifest["format"] == "robogogogo-episode-v2"
        assert len([name for name in archive.namelist() if name.endswith(".png")]) == 6
    checkpoint = client.post("/api/projects", json={"name": "three cameras"}).json()
    capture_names = [
        c["name"]
        for c in client.post(f"/api/projects/{checkpoint['project']['id']}/load", json={}).json()[
            "episode"
        ]["cameras"]
    ]
    assert capture_names == list(images)


@pytest.mark.parametrize(
    "cameras",
    [
        [dict(name="bad_mount", mount="missing", position=[0, 0, 1], look_at=[1, 0, 0])],
        [dict(name="../escape", position=[0, 0, 1], look_at=[1, 0, 0])],
        [dict(name="parallel", position=[0, 0, 1], look_at=[0, 0, 0])],
        [dict(name="duplicate", position=[0, 0, 1], look_at=[1, 0, 0])] * 2,
    ],
)
def test_invalid_camera_rig_does_not_reset_episode(client, cameras):
    before = client.get("/api/snapshot").json()
    assert client.post("/api/vla/reset", json={"cameras": cameras}).status_code == 422
    assert client.get("/api/snapshot").json()["episode"]["id"] == before["episode"]["id"]


def test_head_and_wrist_rig_tracks_physical_mounts(client):
    response = client.post("/api/vla/reset", json={"camera_preset": "head_wrists"})
    response.raise_for_status()
    snapshot = response.json()
    cameras = snapshot["episode"]["cameras"]
    left, right, wrist = cameras
    assert left["mount"] == right["mount"] == "head"
    assert wrist["mount"] == "tool"
    assert np.linalg.norm(np.asarray(left["position"]) - right["position"]) == pytest.approx(0.064)
    assert left["rpy"] == right["rpy"]
    images = {c["name"]: png_observation() for c in cameras}
    observation = client.post(
        "/api/vla/observe", json=dict(images=images, sequence=snapshot["sequence"])
    ).json()
    before = client.app.state.studio.episode.observations[-1]["camera_poses"]
    snapshot = client.post(
        "/api/vla/act",
        json=dict(delta=[0.1, 0, 0, 0, 0, 0], observation=observation["observation"]),
    ).json()
    client.post(
        "/api/vla/observe", json=dict(images=images, sequence=snapshot["sequence"])
    ).raise_for_status()
    after = client.app.state.studio.episode.observations[-1]["camera_poses"]
    assert before["head_left"] == after["head_left"]
    assert before["wrist_1"] != after["wrist_1"]
    saved = client.post("/api/projects", json={"name": "Mounted calibration"}).json()["project"]
    restored = client.post(f"/api/projects/{saved['id']}/load", json={}).json()
    assert restored["episode"]["cameras"] == cameras
