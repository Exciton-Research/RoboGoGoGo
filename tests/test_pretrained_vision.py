"""Optional real-model checks; no network downloads happen during tests or inference."""

import io
import os
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from robogogogo.demo.pretrained import MODEL_ID, ModelNotReady, PretrainedDetector


def test_missing_model_has_an_actionable_error_without_downloading(tmp_path):
    with pytest.raises(ModelNotReady, match="setup-vlm"):
        PretrainedDetector(tmp_path)


@pytest.mark.parametrize("scores", [(0.7, 0.68), (0.21, 0.199)])
def test_ambiguous_model_predictions_are_rejected_before_localization(monkeypatch, scores):
    pytest.importorskip("robogogogo.demo.vision")
    detector = object.__new__(PretrainedDetector)
    detector.detect = lambda rgb, query, **kwargs: [
        {"box": [1, 1, 5, 5], "score": scores[0], "query": query},
        {"box": [6, 1, 10, 5], "score": scores[1], "query": query},
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("Ambiguous predictions reached robot pickup localization")

    monkeypatch.setattr("robogogogo.demo.vision.localize_box", forbidden)
    with pytest.raises(ValueError, match="ambiguous"):
        detector.find(
            "cup",
            rgb=np.zeros((12, 12, 3), np.uint8),
            depth=None,
            position=None,
            rotation=None,
            fovy=45,
        )


@pytest.fixture(scope="module")
def pretrained_setup():
    assets = os.environ.get("ROBOGOGOGO_VLM_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VLM_ASSETS to run cached OWLv2 integration checks")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    environment = pytest.importorskip("robogogogo.demo.environment")
    from robogogogo.demo.objects import CATALOG

    with environment.PandaEnv(assets, object_id="lemon", object_ids=list(CATALOG)) as env:
        yield env, PretrainedDetector(assets)


@pytest.mark.parametrize("moved", [False, True])
def test_real_model_grounds_a_description_without_catalog_assets(
    pretrained_setup, monkeypatch, moved
):
    import mujoco

    env, detector = pretrained_setup
    env.reset(seed=17)
    if moved:
        address = int(env.joints[env.target_index].qposadr[0])
        env.data.qpos[address : address + 2] = [0.30, -0.25]
        for _ in range(250):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
    sensor = env.capture_rgbd()

    def forbidden(*args, **kwargs):
        pytest.fail("Pretrained localization consulted the catalog image matcher")

    monkeypatch.setattr("robogogogo.demo.vision.require_object", forbidden)
    monkeypatch.setattr("robogogogo.demo.vision.CatalogVision.find", forbidden)
    # A specific noun distinguishes the lemon from another citrus fruit (orange).
    result = detector.find("a yellow lemon", **sensor)
    assert result["model"] == MODEL_ID
    assert result["query"] == "a yellow lemon"
    assert result["position"] is not None, result["pickup_error"]
    np.testing.assert_allclose(result["position"], env.data.xpos[env.block], atol=0.014)


def test_real_model_rejects_an_absent_named_object(pretrained_setup):
    env, detector = pretrained_setup
    env.reset(seed=0)
    before = env.capture_state()
    with pytest.raises(ValueError, match="did not confidently detect"):
        detector.find("banana", **env.capture_rgbd())
    np.testing.assert_array_equal(env.capture_state()["integration"], before["integration"])


@pytest.mark.parametrize(
    "name,query", [("lemon", "yellow citrus fruit"), ("rubiks-cube", "Rubik's cube")]
)
def test_pretrained_collection_with_pose_and_contact_labels_blocked(pretrained_setup, name, query):
    from robogogogo.demo.controller import ReferenceController

    env, detector = pretrained_setup
    env.reset(seed=0)
    env.select_target(name)
    result = detector.find(query, **env.capture_rgbd())

    class RobotSensors:
        qpos = env.data.qpos
        site_xpos = env.data.site_xpos

        @property
        def xpos(self):
            pytest.fail("Pretrained controller read object poses")

    robot = SimpleNamespace(
        data=RobotSensors(), site=env.site, dt=env.dt, target=env.target, solve=env.solve
    )
    controller = ReferenceController(robot, camera_pick=result["position"])
    for _ in range(350):
        _, _, done, truncated, info = env.step(controller.action(robot))
        if done or truncated:
            break
        controller.advance(robot)
    assert info["success"], info


def test_pretrained_api_detects_without_motion_and_records_its_action_source(
    pretrained_setup, tmp_path
):
    from fastapi.testclient import TestClient

    from robogogogo.demo.__main__ import create_app

    app = create_app(
        os.environ["ROBOGOGOGO_VLM_ASSETS"], static_dir=tmp_path, recordings=tmp_path / "episodes"
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post(
            "/api/demo/objects/select", json={"identifier": "lemon", "clutter": True}
        ).raise_for_status()
        first = client.post("/api/demo/vision/find", json={"query": "yellow citrus fruit"})
        first.raise_for_status()
        assert first.json()["step_index"] == 0
        image_url = first.json()["vision"]["image_url"]
        assert client.get(image_url).content.startswith(b"\x89PNG")
        second = client.post("/api/demo/vision/find", json={"query": "orange"})
        second.raise_for_status()
        assert second.json()["vision"]["image_url"] != image_url
        assert client.get(image_url).status_code == 422
        missing = client.post(
            "/api/demo/command", json={"command": "run", "perception": "vlm", "query": "banana"}
        )
        assert missing.status_code == 422
        state = client.get("/api/demo/state").json()
        assert state["vision"] is None and state["step_index"] == 0
        client.post(
            "/api/demo/record", json={"command": "start", "images": False}
        ).raise_for_status()
        run = client.post(
            "/api/demo/command",
            json={"command": "run", "perception": "vlm", "query": "yellow citrus fruit"},
        )
        run.raise_for_status()
        assert run.json()["controller_perception"] == "vlm"
        client.post("/api/demo/command", json={"command": "pause"}).raise_for_status()
        saved = client.post("/api/demo/record", json={"command": "save"})
        saved.raise_for_status()
        path = tmp_path / "episodes" / f"{saved.json()['last_saved']['id']}.zip"
        with zipfile.ZipFile(path) as archive:
            with np.load(io.BytesIO(archive.read("trajectory.npz")), allow_pickle=False) as data:
                assert set(data["sources"]) == {"vlm_scripted"}
