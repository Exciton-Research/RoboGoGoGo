import base64

import numpy as np
import pytest

from robogogogo.perception import estimate_stereo, evaluate_depth
from robogogogo.perception.depth import decode_depth


def rig():
    camera = dict(width=192, height=96, vertical_fov_degrees=60.0)
    left = dict(position=[0, 0, 0], rotation=np.eye(3).tolist())
    right = dict(position=[0.064, 0, 0], rotation=np.eye(3).tolist())
    return camera, left, right


def test_shifted_texture_recovers_metric_distance_and_masks_occlusions():
    pytest.importorskip("cv2")
    camera, left_pose, r = rig()
    image = np.random.default_rng(7).integers(0, 256, (96, 192, 3), dtype=np.uint8)
    right = np.zeros_like(image)
    right[:, :-8] = image[:, 8:]
    expected = 96 / (2 * np.tan(np.deg2rad(30))) * 0.064 / 8
    result = estimate_stereo(
        image,
        right,
        camera,
        camera,
        left_pose,
        r,
        ground_truth=np.full((96, 192), expected),
        num_disparities=32,
    )
    assert result.valid.mean() > 0.5
    assert np.nanmedian(result.depth) == pytest.approx(expected, rel=0.01)
    assert not result.valid[:, :16].any()
    metrics = evaluate_depth(result.depth, result.ground_truth)
    assert metrics["mae_m"] < 0.005
    assert 0 < metrics["coverage"] < 1


def test_textureless_stereo_does_not_invent_distance():
    pytest.importorskip("cv2")
    c, left_pose, r = rig()
    image = np.full((96, 192, 3), 128, np.uint8)
    result = estimate_stereo(image, image, c, c, left_pose, r, num_disparities=32)
    assert not result.valid.any()
    assert evaluate_depth(result.depth, np.ones((96, 192)))["mae_m"] is None
    with pytest.raises(ValueError, match="baseline"):
        estimate_stereo(image, image, c, c, left_pose, left_pose, num_disparities=32)
    with pytest.raises(ValueError, match="ordered"):
        estimate_stereo(image, image, c, c, r, left_pose, num_disparities=32)


def test_depth_wire_preserves_nan_and_rejects_corruption():
    camera = dict(width=2, height=1, near=0.01, far=50.0)
    data = np.array([[1.0, np.nan]], dtype="<f4")
    encoded = base64.b64encode(data.tobytes()).decode()
    np.testing.assert_equal(decode_depth(encoded, camera), data)
    for bad in (
        "!!!",
        encoded[:-4],
        base64.b64encode(np.array([-1.0, 1.0], dtype="<f4").tobytes()).decode(),
    ):
        with pytest.raises(ValueError):
            decode_depth(bad, camera)
