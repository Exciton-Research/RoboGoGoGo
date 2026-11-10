from dataclasses import replace

import numpy as np
import pytest

from robogogogo.perception import SceneMemory, SurfaceCloud, backproject, fuse


def image(depth, time, x=0):
    return backproject(
        np.full((9, 9), depth),
        np.full((9, 9, 3), 120),
        np.array([[100.0, 0, 4], [0, 100.0, 4], [0, 0, 1]]),
        np.array([x, 0, 0]),
        np.eye(3),
        timestamp=time,
        uncertainty_m=0.003,
    )


def empty(time):
    return SurfaceCloud(
        np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), time, 0.003, "missing_depth"
    )


def test_occluded_surfaces_persist_and_uncertainty_grows():
    memory = SceneMemory()
    first = memory.observe(image(1.0, 0), target_mask=np.ones(81, bool))
    second = memory.observe(image(0.5, 1))
    assert second.diagnostics["scene_memory"]["remembered"] > 0
    assert second.diagnostics["scene_memory"]["invalidated"] == 0
    assert np.any(second.view_evidence.target)
    assert (
        second.diagnostics["scene_memory"]["max_uncertainty_m"]
        > first.diagnostics["scene_memory"]["max_uncertainty_m"]
    )
    assert np.min(second.view_evidence.confidence) < 1
    # Remembering a target does not turn its hidden flanks into measured grasps.
    assert not second.grasps


def test_measured_free_space_invalidates_old_position_after_motion():
    memory = SceneMemory()
    memory.observe(image(1.0, 0), target_mask=np.ones(81, bool))
    result = memory.observe(image(2.0, 1), target_mask=np.ones(81, bool))
    stats = result.diagnostics["scene_memory"]
    assert stats["invalidated"] > 0 and stats["motion_evidence"] > 0
    assert not any(np.linalg.norm(c["point"] - [0, 0, 1]) < 0.001 for c in memory.cells.values())
    assert np.any(result.view_evidence.points[:, 2] == 2)


def test_missing_depth_ages_memory_without_clearing_or_reusing_proposals():
    memory = SceneMemory(lifetime_s=2)
    first = memory.observe(image(1.0, 0))
    missed = memory.observe(empty(1))
    assert len(missed.view_evidence.points) == len(first.view_evidence.points)
    assert missed.diagnostics["scene_memory"]["invalidated"] == 0
    assert not missed.grasps and not missed.supports
    assert not missed.view_evidence.views
    expired = memory.observe(empty(3))
    assert not len(expired.view_evidence.points)
    assert expired.diagnostics["scene_memory"]["expired"] > 0


def test_reobservation_reduces_uncertainty_but_duplicates_do_not_build_confidence():
    memory = SceneMemory()
    memory.observe(image(1.0, 0))
    memory.observe(empty(1))
    aged = memory.stats["max_uncertainty_m"]
    observed = memory.observe(image(1.0, 2))
    sigma = observed.diagnostics["scene_memory"]["max_uncertainty_m"]
    assert sigma < aged
    memory.observe(image(1.0, 2))
    assert memory.stats["max_uncertainty_m"] >= sigma
    with pytest.raises(ValueError, match="out-of-order"):
        memory.observe(image(1.0, 1))
    assert memory.timestamp == 2
    memory.clear()
    assert memory.timestamp is None and not memory.cells
    memory.observe(image(1.0, 0))


def test_fusion_preserves_each_cameras_actual_measured_rays():
    memory = SceneMemory()
    memory.observe(image(1.0, 0))

    def cloud(center, point):
        points = np.array([point], float)
        return SurfaceCloud(
            points,
            np.zeros_like(points),
            np.zeros_like(points),
            1.0,
            0.003,
            "camera",
            ((dict(position=center, rotation=np.eye(3)), dict(surface_points=points)),),
        )

    merged = fuse([cloud([0, 0, 0], [0.5, 0, 2]), cloud([1, 0, 0], [0, 0, 2])])
    memory.observe(merged)
    assert any(np.linalg.norm(c["point"] - [0, 0, 1]) < 0.001 for c in memory.cells.values())


def test_memory_is_bounded_and_current_observation_proposals_stay_current():
    memory = SceneMemory(capacity=20)
    for index in range(4):
        result = memory.observe(image(1.0, index, x=index))
        assert len(memory.cells) <= 20
        assert result.timestamp == index
    assert any(c["seen"] == 3 for c in memory.cells.values())
    assert any(c["seen"] < 3 for c in memory.cells.values())
    with pytest.raises(ValueError):
        memory.observe(replace(image(1.0, 4), uncertainty_m=float("nan")))
    assert memory.timestamp == 3


def test_aging_memory_increases_incentive_to_revisit_same_view():
    memory = SceneMemory()
    cloud = image(1.0, 0)
    current = memory.observe(cloud)
    pose, camera = cloud.views[0]
    remembered = memory.observe(empty(1))
    fresh_gain = current.view_evidence.score(pose, pose, camera)["gain"]
    stale_gain = remembered.view_evidence.score(pose, pose, camera)["gain"]
    assert fresh_gain == 0
    assert stale_gain > fresh_gain


def test_motion_evidence_inflates_uncertainty_of_still_hidden_target_parts():
    memory = SceneMemory()
    first = image(1.0, 0)
    memory.observe(first, target_mask=np.ones(len(first.points), bool))
    # Only one valid center ray looks through the old target location.
    depth = np.full((9, 9), np.nan)
    depth[4, 4] = 2.0
    later = backproject(
        depth,
        np.full((9, 9, 3), 120),
        np.array([[100.0, 0, 4], [0, 100.0, 4], [0, 0, 1]]),
        np.zeros(3),
        np.eye(3),
        timestamp=1.0,
        uncertainty_m=0.003,
    )
    result = memory.observe(later, target_mask=np.ones(1, bool))
    assert result.diagnostics["scene_memory"]["motion_evidence"] > 0
    hidden = [c for c in memory.cells.values() if c["seen"] == 0]
    assert hidden and min(c["variance"] for c in hidden) >= 0.02**2
    assert np.min(result.view_evidence.confidence) < 0.5
