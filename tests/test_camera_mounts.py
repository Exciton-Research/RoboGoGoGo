"""Physical mount transforms and optical camera conventions."""

import numpy as np
import pytest

from robogogogo.studio.schemas import CameraSpec
from robogogogo.vla.cameras import world_camera_pose


def test_camera_transform_composes_offset_orientation_and_parent():
    camera = CameraSpec(
        name="wrist", mount="tool", position=(0.1, 0.02, 0), rpy=(0, 0, 0)
    ).model_dump()
    camera["body"] = 0
    state = dict(positions=[[2, 3, 1]], rotations=[[[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    pose = world_camera_pose(camera, state)
    np.testing.assert_allclose(pose["position"], [1.98, 3.1, 1])
    rotation = np.asarray(pose["rotation"])
    np.testing.assert_allclose(rotation[:, 2], [0, 1, 0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.mark.parametrize(
    "rpy,forward",
    [((0, 0, 0), (1, 0, 0)), ((0, np.pi / 2, 0), (0, 0, -1)), ((0, 0, np.pi / 2), (0, 1, 0))],
)
def test_mount_angles_define_view_direction(rpy, forward):
    camera = CameraSpec(name="head", position=(0, 0, 0), rpy=rpy).model_dump()
    np.testing.assert_allclose(
        np.asarray(world_camera_pose(camera, {})["rotation"])[:, 2], forward, atol=1e-12
    )


def test_camera_requires_orientation_and_finite_angles():
    with pytest.raises(ValueError):
        CameraSpec(name="head", position=(0, 0, 0))
    with pytest.raises(ValueError):
        CameraSpec(name="head", position=(0, 0, 0), rpy=(0, float("nan"), 0))
