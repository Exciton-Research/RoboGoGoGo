"""Sensor-only proposals, missing-data guards, and integration with recovery search."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robogogogo.perception.recovery import SurfaceCloud, detect, foreground, fuse
from robogogogo.planning.motion import PlanningError
from robogogogo.recovery_planning import RecoveryPlanner
from robogogogo.studio.schemas import PlanTask


def example(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "examples" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMERA, RIG = example("camera_recovery"), example("recovery_search")


def test_rgbd_multiview_detects_opposing_grasp_surfaces_and_landing_patch():
    cloud = CAMERA.observe()
    result = detect(cloud, seed=[0.2, -0.09, 0.5])
    assert len(result.grasps) > 2
    assert result.supports
    assert result.source == "synthetic_rgbd"
    assert all(abs(g["position"][0] - 0.2) < 0.005 for g in result.grasps)
    assert all(abs(s["position"][2] - 0.45) < 0.003 for s in result.supports)
    assert all(g["evidence_points"] >= 6 for g in result.grasps)


def test_detection_follows_pixels_when_scene_is_translated():
    offset = np.array([0.12, -0.08, 0.07])
    result = detect(CAMERA.observe(offset), seed=np.array([0.2, -0.09, 0.5]) + offset)
    assert result.grasps and result.supports
    assert all(abs(g["position"][0] - 0.32) < 0.005 for g in result.grasps)
    assert all(abs(s["position"][2] - 0.52) < 0.003 for s in result.supports)


def test_missing_opposing_flank_does_not_invent_grasp():
    cloud = CAMERA.observe()
    mask = foreground(cloud, [0.2, -0.09, 0.5])
    keep = ~mask | (cloud.points[:, 0] < 0.199)
    partial = SurfaceCloud(
        cloud.points[keep], cloud.normals[keep], cloud.colors[keep], 0.0, 0.002, "partial"
    )
    result = detect(partial, target_mask=mask[keep])
    assert not result.grasps


def test_unknown_holes_and_obstacles_reject_landing_footprints():
    x, y = np.meshgrid(np.arange(-0.1, 0.101, 0.005), np.arange(-0.1, 0.101, 0.005))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.4)]

    def cloud(p):
        return SurfaceCloud(
            p, np.tile([0, 0, 1.0], (len(p), 1)), np.zeros_like(p), 0.0, 0.002, "fixture"
        )

    assert detect(cloud(points)).supports
    # A cross-shaped occlusion leaves no complete minimum footprint in any quadrant.
    visible = points[(abs(points[:, 0]) > 0.025) & (abs(points[:, 1]) > 0.025)]
    assert not detect(cloud(visible)).supports
    obstacles = points.copy()
    obstacles[:, 2] += 0.1
    occupied = SurfaceCloud(
        np.r_[points, obstacles],
        np.r_[np.tile([0, 0, 1.0], (len(points), 1)), np.zeros_like(obstacles)],
        np.zeros((len(points) * 2, 3)),
        0.0,
        0.002,
        "fixture",
    )
    assert not detect(occupied).supports


def test_camera_planner_uses_observed_grasps_and_rejects_stale_or_empty_views():
    session, grasp, world = RIG.fixture()
    grasp.grippers["left"]["holding"] = True
    grasp.enable_recovery(camera_based=True)
    planner = grasp.recovery.planner
    grasp.observe_recovery(CAMERA.observe(), seed=[0.2, -0.09, 0.5])
    route = planner.helper_path(grasp, session, world, "left")
    assert any(np.allclose(route.grasp_position, g["position"]) for g in planner.observation.grasps)
    np.testing.assert_array_equal(route.joints[-1][:3], session.state.q[:3])
    session.elapsed = 0.6
    with pytest.raises(PlanningError, match="fresh calibrated"):
        planner.helper_path(grasp, session, world, "left")
    session.elapsed = 0.0
    planner.observation = detect(
        CAMERA.observe(), target_mask=np.zeros(len(CAMERA.observe().points), bool)
    )
    with pytest.raises(PlanningError, match="No camera grasp"):
        planner.helper_path(grasp, session, world, "left")


def test_camera_support_positions_are_not_snapped_to_registry_centers():
    session, grasp, _ = RIG.fixture()
    planner = RecoveryPlanner(session.model, grasp.arms)
    planner.vision_required = True
    planner.observation = detect(CAMERA.observe(), seed=[0.2, -0.09, 0.5])
    grasp.supports = np.array([[0.48, 0, 0.125, 0.20, 0.45]])
    candidates = planner.support_candidates(grasp, session)
    assert candidates
    assert any(not np.allclose(s[:2], grasp.supports[i, :2]) for i, s in candidates)
    grasp.supports = np.empty((0, 5))
    with pytest.raises(PlanningError, match="verifiable physical support"):
        planner.support_candidates(grasp, session)


def test_synchronization_and_camera_configuration_guards():
    cloud = CAMERA.observe()
    late = SurfaceCloud(cloud.points, cloud.normals, cloud.colors, 0.1, 0.002, "late")
    with pytest.raises(ValueError, match="synchronized"):
        fuse([cloud, late])
    with pytest.raises(ValueError, match="coordinated recovery"):
        PlanTask(camera_recovery=True)


def test_studio_uses_rgb_only_and_records_the_proposals():
    from types import SimpleNamespace

    cv = pytest.importorskip("cv2")
    from robogogogo.studio.perception import update_recovery_observation

    image = np.random.default_rng(19).integers(0, 256, (96, 192, 3), dtype=np.uint8)
    right = np.zeros_like(image)
    right[:, :-24] = image[:, 24:]
    cameras = [
        dict(name=n, width=192, height=96, vertical_fov_degrees=60.0)
        for n in ("head_left", "head_right")
    ]
    poses = {
        n: dict(position=[x, 0, 0], rotation=np.eye(3).tolist())
        for n, x in [("head_left", 0.0), ("head_right", 0.064)]
    }
    frames = {
        n: cv.imencode(".png", cv.cvtColor(img, cv.COLOR_RGB2BGR))[1].tobytes()
        for n, img in [("head_left", image), ("head_right", right)]
    }
    observation = dict(time=0.0, camera_poses=poses)
    # Deliberately no renderer depths, material, object state, or support registry.
    planner = SimpleNamespace(observation=None)
    m = SimpleNamespace(recovery=SimpleNamespace(planner=planner), grippers={})
    workspace = SimpleNamespace(
        task_service=SimpleNamespace(manipulation=m),
        session=None,
        episode=SimpleNamespace(cameras=cameras, images=[frames], observations=[observation]),
    )
    update_recovery_observation(workspace, 0)
    assert planner.observation.source == "calibrated_rgb_stereo"
    assert planner.observation.diagnostics["visible_points"] > 0
    assert observation["recovery_perception"]["supports"]
    assert not planner.observation.grasps
    workspace.episode.cameras = []
    update_recovery_observation(workspace, 0)
    assert not planner.observation.supports
    assert "head_left" in planner.observation.diagnostics["error"]


def test_rectified_stereo_extrinsics_backproject_in_world_coordinates():
    pytest.importorskip("cv2")
    from robogogogo.perception.recovery import backproject
    from robogogogo.perception.stereo import estimate_stereo

    camera = dict(width=192, height=96, vertical_fov_degrees=60.0)
    rotation = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    center = np.array([0.1, 0.2, 0.3])
    poses = [
        dict(position=(center + rotation @ [x, 0, 0]).tolist(), rotation=rotation.tolist())
        for x in (0.0, 0.064)
    ]
    image = np.random.default_rng(8).integers(0, 256, (96, 192, 3), dtype=np.uint8)
    right = np.zeros_like(image)
    right[:, :-24] = image[:, 24:]
    result = estimate_stereo(image, right, camera, camera, *poses, num_disparities=32)
    cloud = backproject(
        result.depth,
        result.rectified_left,
        result.intrinsics,
        result.position,
        result.rotation,
        timestamp=0.0,
    )
    expected_depth = result.focal_px * result.baseline_m / 24
    assert np.median(cloud.points[:, 0]) == pytest.approx(0.1 + expected_depth, abs=0.003)


@pytest.mark.parametrize("scenario,phase", [("helper", "secured"), ("support", "placed")])
def test_camera_proposals_drive_physical_recovery(scenario, phase):
    from functools import partial

    factory = (
        partial(
            RIG.fixture,
            supports=[[0.2, 0, 0.12, 0.2, 0.48], [0.48, 0, 0.12, 0.2, 0.45]],
            clearance_m=0.005,
            finger_tip=0.205,
        )
        if scenario == "support"
        else RIG.fixture
    )
    result = RIG.run(scenario, camera_provider=CAMERA.physical_sensor, fixture_factory=factory)
    assert result["failure"] is None
    assert result["recovery"]["phase"] == phase
    assert result["recovery"]["camera_mode"]
    assert result["recovery"]["search_checks"] > 0
    assert not result["object"]["material"]["fractured"]
    if scenario == "helper":
        assert all(g["holding"] for g in result["grippers"].values())
    else:
        index = result["recovery"]["support_index"]
        assert result["support_load_n"][index + 1] > 0.8 * 0.16 * 9.81
        assert result["object"]["material"]["damage_fraction"] == pytest.approx(0.02)
        assert all(g["command"] == "open" for g in result["grippers"].values())
