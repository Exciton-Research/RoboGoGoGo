"""Pixel localization, abstention, and the camera controller's sensor boundary."""

# Check optional demo dependencies before importing their project modules.
# ruff: noqa: E402

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from robogogogo.demo.controller import ReferenceController
from robogogogo.demo.environment import PandaEnv
from robogogogo.demo.objects import CATALOG
from robogogogo.demo.vision import CatalogVision, point_cloud


def test_depth_projection_uses_pixel_centers_and_camera_calibration():
    cloud = point_cloud(np.ones((3, 3)), np.array([1, 2, 3]), np.eye(3), 90)
    np.testing.assert_allclose(cloud[1, 1], [1, 2, 2])
    assert cloud[0, 0, 0] < 1 and cloud[0, 0, 1] > 2


@pytest.fixture(scope="module")
def camera_setup():
    assets = os.environ.get("ROBOGOGOGO_VISION_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VISION_ASSETS to run native RGB-D integration checks")
    with PandaEnv(assets, object_id="lemon", object_ids=list(CATALOG)) as env:
        yield env, CatalogVision(assets)


@pytest.mark.parametrize("shuffle", [False, True])
def test_camera_finds_all_names_after_objects_change_positions(camera_setup, shuffle):
    env, vision = camera_setup
    env.reset(seed=17)
    if shuffle:
        order = np.random.default_rng(23).permutation(8)
        for index, joint in enumerate(env.joints):
            address = int(joint.qposadr[0])
            slot = order[index]
            env.data.qpos[address : address + 2] = [
                0.28 + slot % 4 * 0.15,
                -0.28 + slot // 4 * 0.19,
            ]
        for _ in range(250):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
    sensor = env.capture_rgbd()
    for index, name in enumerate(CATALOG):
        match = vision.find(name, **sensor, catalog_ids=list(CATALOG))
        # Privileged state is used only as an evaluation label, never by the matcher.
        expected = env.data.xpos[env.bodies[index], :2]
        assert np.linalg.norm(match["position"][:2] - expected) < 0.012, name


def test_absent_and_occluded_targets_stop_instead_of_using_their_scene_pose(camera_setup):
    env, vision = camera_setup
    env.reset(seed=0)
    address = int(env.joints[list(CATALOG).index("lemon")].qposadr[0])
    env.data.qpos[address : address + 3] = [2, 2, 0.8]
    mujoco.mj_forward(env.model, env.data)
    sensor = env.capture_rgbd()
    with pytest.raises(ValueError, match="identify|ambiguous"):
        vision.find("lemon", **sensor, catalog_ids=list(CATALOG))
    sensor["rgb"][:] = 0
    with pytest.raises(ValueError, match="identify|ambiguous"):
        vision.find("orange", **sensor, catalog_ids=list(CATALOG))


@pytest.mark.parametrize("name", list(CATALOG))
def test_camera_collection_cannot_read_object_poses_or_labeled_contacts(camera_setup, name):
    env, vision = camera_setup
    env.reset(seed=0)
    env.select_target(name)
    match = vision.find(name, **env.capture_rgbd(), catalog_ids=list(CATALOG))

    class RobotSensors:
        @property
        def xpos(self):
            raise AssertionError("Camera controller read privileged body positions")

        qpos = env.data.qpos
        site_xpos = env.data.site_xpos

    controller_env = SimpleNamespace(
        data=RobotSensors(),
        site=env.site,
        dt=env.dt,
        target=env.target,
        solve=env.solve,
    )
    controller = ReferenceController(controller_env, camera_pick=match["position"])
    for _ in range(350):
        _, _, ended, truncated, info = env.step(controller.action(controller_env))
        if ended or truncated:
            break
        controller.advance(controller_env)
    assert info["success"], info


def test_camera_api_returns_detection_image_and_clears_it_on_reset(camera_setup, tmp_path):
    from fastapi.testclient import TestClient

    from robogogogo.demo.__main__ import create_app

    assets = os.environ["ROBOGOGOGO_VISION_ASSETS"]
    with TestClient(
        create_app(assets, static_dir=tmp_path, recordings=tmp_path / "episodes"),
        base_url="http://127.0.0.1",
    ) as client:
        client.post(
            "/api/demo/objects/select", json={"identifier": "lemon", "clutter": True}
        ).raise_for_status()
        started = client.post("/api/demo/command", json={"command": "run", "perception": "camera"})
        started.raise_for_status()
        assert started.json()["controller_perception"] == "camera"
        detection = started.json()["vision"]
        assert detection["name"] == "Lemon"
        image = client.get(detection["image_url"])
        assert image.headers["content-type"] == "image/png"
        assert image.content.startswith(b"\x89PNG")
        client.post("/api/demo/command", json={"command": "pause"}).raise_for_status()
        reset = client.post("/api/demo/reset", json={"seed": 0})
        reset.raise_for_status()
        assert reset.json()["vision"] is None
        assert client.get(detection["image_url"]).status_code == 422
        reference = client.post("/api/demo/command", json={"command": "run"})
        reference.raise_for_status()
        assert reference.json()["controller_perception"] == "simulator"
        assert reference.json()["vision"] is None
        client.post("/api/demo/command", json={"command": "pause"}).raise_for_status()
