"""Optional real-model checks; no network downloads happen during tests or inference."""

import io
import os
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from robogogogo.demo.pretrained import MODEL_ID, ModelNotReady, PretrainedDetector


def test_new_household_object_is_excluded_from_catalog_image_matching(tmp_path):
    vision = pytest.importorskip("robogogogo.demo.vision")
    from robogogogo.demo.objects import CATALOG, DEFAULT_CLUTTER, clutter_objects

    assert "sugar-box" not in DEFAULT_CLUTTER
    assert not CATALOG["sugar-box"]["reference_matching"]
    assert clutter_objects("sugar-box") == ["sugar-box", *DEFAULT_CLUTTER]
    with pytest.raises(ValueError, match="excluded"):
        vision.CatalogVision(tmp_path)._reference("sugar-box")


def test_missing_model_has_an_actionable_error_without_downloading(tmp_path):
    with pytest.raises(ModelNotReady, match="setup-vlm"):
        PretrainedDetector(tmp_path)


@pytest.mark.parametrize("scores", [(0.7, 0.68), (0.21, 0.199)])
@pytest.mark.parametrize("crop", [None, [0, 0, 12, 12]])
def test_ambiguous_model_predictions_are_rejected_before_localization(monkeypatch, scores, crop):
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
            crop=crop,
        )


def test_cropped_detections_are_localized_in_the_original_camera_frame(monkeypatch):
    pytest.importorskip("robogogogo.demo.vision")
    detector = object.__new__(PretrainedDetector)
    detector.device = "test"
    rgb = np.zeros((40, 50, 3), np.uint8)
    depth = np.ones((40, 50))

    def detect(image, query, **kwargs):
        assert image.shape == (20, 30, 3)
        return [{"box": [2, 3, 12, 15], "score": 0.7, "query": query}]

    def localize(box, **sensor):
        assert box == [12, 8, 22, 20]
        assert sensor["rgb"] is rgb and sensor["depth"] is depth
        return np.array([0.4, -0.2, 0.8]), box, np.array([0.4, -0.2, 0.82])

    detector.detect = detect
    monkeypatch.setattr("robogogogo.demo.vision.localize_box", localize)
    result = detector.find(
        "can",
        rgb=rgb,
        depth=depth,
        position=None,
        rotation=None,
        fovy=45,
        crop=[10, 5, 40, 25],
    )
    assert result["box"] == [12, 8, 22, 20]
    assert result["position"] == [0.4, -0.2, 0.8]


@pytest.fixture(scope="module")
def pretrained_setup():
    assets = os.environ.get("ROBOGOGOGO_VLM_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VLM_ASSETS to run cached OWLv2 integration checks")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    environment = pytest.importorskip("robogogogo.demo.environment")
    from robogogogo.demo.objects import clutter_objects

    with environment.PandaEnv(assets, object_id="lemon", object_ids=clutter_objects()) as env:
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
    controller = ReferenceController(
        robot, camera_pick=result["position"], camera_grasp=result["grasp_position"]
    )
    for _ in range(350):
        _, _, done, truncated, info = env.step(controller.action(robot))
        if done or truncated:
            break
        controller.advance(robot)
    assert info["success"], info


def test_new_household_scan_is_detected_and_collected_without_catalog_references(
    pretrained_setup, monkeypatch, tmp_path
):
    from robogogogo.demo.controller import ReferenceController
    from robogogogo.demo.environment import PandaEnv
    from robogogogo.demo.objects import clutter_objects
    from robogogogo.demo.recording import Episode, Recording

    _, detector = pretrained_setup
    assets = os.environ["ROBOGOGOGO_VLM_ASSETS"]
    identifiers = clutter_objects("sugar-box")
    with PandaEnv(assets, object_id="sugar-box", object_ids=identifiers) as env:
        assert len(env.objects) == 9
        assert env.model.body_mass[env.block] == pytest.approx(0.25)
        original = env.data.xpos[env.bodies].copy()
        env.reset(seed=1)
        shuffled = env.data.xpos[env.bodies].copy()
        assert (
            np.count_nonzero(np.linalg.norm(shuffled[:, :2] - original[:, :2], axis=1) > 0.10) >= 6
        )
        env.reset(seed=1)
        np.testing.assert_allclose(env.data.xpos[env.bodies], shuffled)
        sensor = env.capture_rgbd(view="side")

        def forbidden(*args, **kwargs):
            pytest.fail("New object recognition consulted catalog references")

        monkeypatch.setattr("robogogogo.demo.vision.require_object", forbidden)
        monkeypatch.setattr("robogogogo.demo.vision.CatalogVision.find", forbidden)
        detection = detector.find("tall yellow box", **sensor)
        assert detection["model"] == MODEL_ID
        assert detection["grasp_position"][2] > detection["position"][2] + 0.03

        class RobotSensors:
            qpos = env.data.qpos
            site_xpos = env.data.site_xpos

            @property
            def xpos(self):
                pytest.fail("New-object controller read privileged object poses")

        robot = SimpleNamespace(
            data=RobotSensors(), site=env.site, dt=env.dt, target=env.target, solve=env.solve
        )
        controller = ReferenceController(
            robot, camera_pick=detection["position"], camera_grasp=detection["grasp_position"]
        )
        recording = Recording(tmp_path, env, seed=1, images=False)
        for _ in range(350):
            action = controller.action(robot)
            result = env.step(action)
            recording.append(env, action, result, "vlm_scripted")
            if result[2] or result[3]:
                break
            controller.advance(robot)
        assert result[4]["success"], result[4]
        # The other eight objects remain on the source table after collection.
        assert np.max(np.linalg.norm(env.data.xpos[env.bodies[1:]] - shuffled[1:], axis=1)) < 0.02
        recording.save(env, "environment_ended")
        path = tmp_path / f"{recording.id}.zip"
    with PandaEnv(assets, object_id="sugar-box", object_ids=identifiers) as replay:
        episode = Episode(path, replay)
        replay.restore_state(episode.state(episode.count))
        assert replay.info()["success"]


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


def test_default_soup_name_retries_the_camera_and_collects_from_full_clutter(
    pretrained_setup, monkeypatch, tmp_path
):
    import time

    from fastapi.testclient import TestClient

    from robogogogo.demo.__main__ import create_app

    _, detector = pretrained_setup
    monkeypatch.setattr("robogogogo.demo.session.PretrainedDetector", lambda _: detector)
    app = create_app(
        os.environ["ROBOGOGOGO_VLM_ASSETS"], static_dir=tmp_path, recordings=tmp_path / "episodes"
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post(
            "/api/demo/objects/select",
            json={
                "identifier": "sugar-box",
                "clutter": True,
                "seed": 1,
            },
        ).raise_for_status()
        client.post("/api/demo/target", json={"identifier": "soup-can"}).raise_for_status()
        before = client.get("/api/demo/state").json()
        assert len(before["instances"]) == 9

        def forbidden(*args, **kwargs):
            pytest.fail("Camera retry consulted catalog references")

        monkeypatch.setattr("robogogogo.demo.vision.require_object", forbidden)
        monkeypatch.setattr("robogogogo.demo.vision.CatalogVision.find", forbidden)
        found = client.post("/api/demo/vision/find", json={"query": "Tomato soup can"})
        found.raise_for_status()
        state = found.json()
        assert state["step_index"] == 0
        detection = state["vision"]
        assert detection["camera_view"] == "front"
        assert len(detection["detection_crop"]) == 4
        assert detection["query"] == "Tomato soup can"
        assert detection["score"] >= 0.2 and detection["margin"] >= 0.06
        np.testing.assert_allclose(detection["position"], before["block"], atol=0.014)
        client.post(
            "/api/demo/record", json={"command": "start", "images": False}
        ).raise_for_status()
        client.post(
            "/api/demo/command",
            json={
                "command": "run",
                "perception": "vlm",
                "query": "Tomato soup can",
            },
        ).raise_for_status()
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            state = client.get("/api/demo/state").json()
            if state["status"] != "running":
                break
            time.sleep(0.1)
        assert state["status"] == "complete", state["message"]
        assert state["last_saved"]["outcome"] == "success"
        # Translation must clear the tall carton while the can stays held.
        path = tmp_path / "episodes" / f"{state['last_saved']['id']}.zip"
        with zipfile.ZipFile(path) as archive:
            with np.load(io.BytesIO(archive.read("trajectory.npz")), allow_pickle=False) as data:
                assert set(data["sources"]) == {"vlm_scripted"}
                # Soup is the second free joint: qpos follows 9 robot coordinates.
                centers = data["qpos"][:, 16:19]
                assert np.min(centers[115:155, 2]) > 0.99
                # The can remains near the fingers until the release phase;
                # landing in the tray after an early drop is not sufficient.
                tools = data["proprio"][:, 18:21]
                assert np.max(np.linalg.norm(tools[115:202] - centers[115:202], axis=1)) < 0.065
