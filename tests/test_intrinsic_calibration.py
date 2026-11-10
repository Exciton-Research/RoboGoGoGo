import numpy as np
import pytest

from robogogogo.hardware.calibrate_intrinsics import fit_corners


def measurements():
    cv = pytest.importorskip("cv2")
    board = np.zeros((54, 3), np.float32)
    board[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 0.025
    k = np.array([[600.0, 0, 320], [0, 610.0, 240], [0, 0, 1]])
    views = []
    for i in range(16):
        rotation = np.array([0.15 * ((i % 4) - 1.5), 0.15 * ((i // 4) - 1.5), 0.03 * i])
        translation = np.array(
            [-0.18 + 0.06 * (i % 4), -0.14 + 0.06 * (i // 4), 0.65 + 0.07 * (i % 3)]
        )
        points, _ = cv.projectPoints(board, rotation, translation, k, np.zeros(5))
        views.append(points)
    return k, views


def test_intrinsics_are_measured_with_held_out_views():
    expected, points = measurements()
    result = fit_corners(points, (640, 480), (9, 6), 0.025)
    np.testing.assert_allclose(result["intrinsics"], expected, atol=0.05)
    assert len(result["held_out_indices"]) == 4
    assert result["held_out_max_rms_px"] < 0.01


def test_repeated_or_insufficient_board_poses_are_rejected():
    _, points = measurements()
    with pytest.raises(ValueError, match="14"):
        fit_corners(points[:3], (640, 480), (9, 6), 0.025)
    with pytest.raises(ValueError, match="Move the board"):
        fit_corners([points[0]] * 16, (640, 480), (9, 6), 0.025)


def test_bad_held_out_corner_measurements_fail_quality_check():
    _, points = measurements()
    points[0] = points[0] + np.random.default_rng(17).normal(0, 8, points[0].shape)
    with pytest.raises(ValueError, match="reprojection"):
        fit_corners(points, (640, 480), (9, 6), 0.025)
