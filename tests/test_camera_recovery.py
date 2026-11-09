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
    planner = RecoveryPlanner(session.model, grasp.arms)
    planner.vision_required = True
    planner.observation = detect(CAMERA.observe(), seed=[0.2, -0.09, 0.5])
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
