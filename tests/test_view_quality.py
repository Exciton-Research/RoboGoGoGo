"""Visibility and uncertainty tests independent of the robot's collision model."""

import numpy as np

from robogogogo.perception import SurfaceCloud, detect
from robogogogo.perception.view_quality import ViewEvidence


def pose(x=0, z=0):
    return dict(position=np.array([x, 0, z]), rotation=np.eye(3))


def test_observed_occluder_blocks_ray_but_side_view_reveals_target():
    yy, xx = np.mgrid[-0.08:0.081:0.006, -0.08:0.081:0.006]
    panel = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.5)))
    evidence = ViewEvidence(
        panel, np.tile([0, 0, 1], (len(panel), 1)), np.zeros(len(panel), bool), 0.005
    )
    point, normal = np.array([[0.0, 0, 1]]), np.array([[0.0, 0, 1]])
    hidden, visibility = evidence.quality(pose(), {}, point, normal)
    revealed, side_visibility = evidence.quality(pose(x=0.3), {}, point, normal)
    assert not visibility[0] and hidden[0] == 0
    assert side_visibility[0] and revealed[0] > 0


def test_uncertainty_range_incidence_and_fov_affect_quality():
    point, normal = np.array([[0.0, 0, 1]]), np.array([[0.0, 0, 1]])
    evidence = ViewEvidence(point, normal, np.ones(1, bool), 0.005)
    far, _ = evidence.quality(pose(), {}, point, normal)
    near, _ = evidence.quality(pose(z=0.5), {}, point, normal)
    grazing, _ = evidence.quality(pose(), {}, point, np.array([[1.0, 0, 0]]))
    uncertain = ViewEvidence(point, normal, np.ones(1, bool), 0.02)
    noisy, _ = uncertain.quality(pose(), {}, point, normal)
    outside, _ = evidence.quality(pose(x=10), {}, point, normal)
    assert near[0] > far[0] > grazing[0]
    assert far[0] > noisy[0] and outside[0] == 0
    assert evidence.score(pose(), pose(), {})["gain"] == 0
    assert evidence.score(pose(), pose(z=0.5), {})["gain"] > 0


def test_empty_evidence_and_unknown_probes_never_create_action_proposals():
    cloud = SurfaceCloud(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), 0.0, 0.005, "test")
    observation = detect(cloud)
    assert observation.view_evidence.score(pose(), pose(x=0.2), {})["gain"] == 0
    assert not observation.grasps and not observation.supports
    assert "view_evidence" not in observation.summary()


def test_synchronized_reference_camera_discounts_already_observed_view():
    point, normal = np.array([[0.0, 0, 1]]), np.array([[0.0, 0, 1]])
    fresh = ViewEvidence(point, normal, np.ones(1, bool), 0.005)
    covered = ViewEvidence(point, normal, np.ones(1, bool), 0.005, ((pose(z=0.5), {}),))
    assert fresh.score(pose(), pose(z=0.5), {})["gain"] > 0
    assert covered.score(pose(), pose(z=0.5), {})["gain"] == 0
