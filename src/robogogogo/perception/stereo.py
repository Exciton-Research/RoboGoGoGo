"""Calibrated pinhole stereo with explicit invalid pixels and metric optical depth."""

from dataclasses import dataclass

import numpy as np


def opencv():
    try:
        import cv2

        return cv2
    except ImportError as error:
        raise ValueError(
            'Install stereo support with pip install "robogogogo[perception]"'
        ) from error


@dataclass(frozen=True)
class StereoResult:
    depth: np.ndarray
    disparity: np.ndarray
    valid: np.ndarray
    rectified_left: np.ndarray
    rectified_right: np.ndarray
    ground_truth: np.ndarray | None
    baseline_m: float
    focal_px: float


def intrinsic(camera):
    if camera.get("intrinsics") is not None:
        matrix = np.asarray(camera["intrinsics"], dtype=float)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(matrix[0, 1]) > 1e-10
            or abs(matrix[1, 0]) > 1e-10
        ):
            raise ValueError("Invalid pinhole intrinsics (zero skew required)")
        return matrix
    f = camera["height"] / (2 * np.tan(np.deg2rad(camera["vertical_fov_degrees"]) / 2))
    return np.array(
        [[f, 0, camera["width"] / 2], [0, f, camera["height"] / 2], [0, 0, 1]], dtype=float
    )


def estimate_stereo(
    left,
    right,
    left_camera,
    right_camera,
    left_pose,
    right_pose,
    *,
    ground_truth=None,
    num_disparities=64,
    block_size=5,
):
    """Rectify synchronized uint8 RGB images, then estimate depth via StereoSGBM.

    Extrinsics are optical camera-to-world. Output depth is Z in the rectified
    left camera, in meters; NaN means unknown. Only horizontal, positive-disparity
    stereo rigs are supported (swap cameras for a reversed rig). Pinhole lenses;
    real distorted images must be undistorted/calibrated before using this API.
    Ground truth is evaluation-only and never enters the matcher.
    """
    cv = opencv()
    left, right = np.asarray(left), np.asarray(right)
    h, w = left_camera["height"], left_camera["width"]
    if (
        left.shape != (h, w, 3)
        or right.shape != left.shape
        or left.dtype != np.uint8
        or right.dtype != np.uint8
        or (right_camera["height"], right_camera["width"]) != (h, w)
    ):
        raise ValueError("Stereo requires equal-size uint8 RGB camera frames")
    if (
        num_disparities < 16
        or num_disparities % 16
        or num_disparities >= w
        or block_size < 3
        or block_size % 2 == 0
        or block_size > min(h, w)
    ):
        raise ValueError(
            "Use a disparity range divisible by 16 and smaller than width; odd block size >= 3"
        )
    kl, kr = intrinsic(left_camera), intrinsic(right_camera)
    rotations, centers = [], []
    for pose in (left_pose, right_pose):
        r, c = np.asarray(pose["rotation"], dtype=float), np.asarray(pose["position"], dtype=float)
        if (
            r.shape != (3, 3)
            or c.shape != (3,)
            or not np.isfinite(r).all()
            or not np.isfinite(c).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5)
        ):
            raise ValueError("Invalid optical camera pose")
        rotations.append(r)
        centers.append(c)
    rotation = rotations[1].T @ rotations[0]
    translation = rotations[1].T @ (centers[0] - centers[1])
    baseline = float(np.linalg.norm(translation))
    if baseline < 1e-5:
        raise ValueError("Stereo cameras must have a nonzero baseline")
    r1, r2, p1, p2, q, roi1, roi2 = cv.stereoRectify(
        kl, None, kr, None, (w, h), rotation, translation, flags=cv.CALIB_ZERO_DISPARITY, alpha=0
    )
    if abs(p2[1, 3]) > abs(p2[0, 3]) or p2[0, 3] >= -1e-8:
        raise ValueError("Use a horizontal stereo pair ordered left then right")
    maps = [
        cv.initUndistortRectifyMap(k, None, r, p, (w, h), cv.CV_32FC1)
        for k, r, p in ((kl, r1, p1), (kr, r2, p2))
    ]
    rect_left, rect_right = [
        cv.remap(img, *m, cv.INTER_LINEAR) for img, m in zip((left, right), maps)
    ]
    gl, gr = [cv.cvtColor(img, cv.COLOR_RGB2GRAY) for img in (rect_left, rect_right)]

    def matcher(minimum):
        return cv.StereoSGBM_create(
            minDisparity=minimum,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            uniquenessRatio=12,
            speckleWindowSize=50,
            speckleRange=2,
            disp12MaxDiff=1,
        )

    d = matcher(0).compute(gl, gr).astype(np.float32) / 16
    dr = matcher(-num_disparities).compute(gr, gl).astype(np.float32) / 16
    yy, xx = np.indices((h, w))
    xr = np.rint(xx - d).astype(int)
    valid = (d > 0.5) & (xr >= 0) & (xr < w)
    valid &= np.abs(d + dr[yy, np.clip(xr, 0, w - 1)]) <= 1
    mean = cv.boxFilter(gl.astype(np.float32), -1, (block_size, block_size))
    variance = cv.boxFilter(gl.astype(np.float32) ** 2, -1, (block_size, block_size)) - mean**2
    valid &= variance > 4
    for roi, xcoord in ((roi1, xx), (roi2, xr)):
        x, y, rw, rh = roi
        valid &= (xcoord >= x) & (xcoord < x + rw) & (yy >= y) & (yy < y + rh)
    depth = cv.reprojectImageTo3D(d, q)[..., 2]
    valid &= np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, np.nan).astype(np.float32)
    truth = None
    if ground_truth is not None:
        gt = np.asarray(ground_truth, dtype=np.float32)
        if gt.shape != (h, w) or np.isinf(gt).any() or (gt[np.isfinite(gt)] <= 0).any():
            raise ValueError("Ground truth must be positive optical Z meters or NaN")
        # Rotate original rays into rectified camera; remapping Z alone is incorrect.
        rays = np.stack(((xx - kl[0, 2]) / kl[0, 0], (yy - kl[1, 2]) / kl[1, 1], np.ones((h, w))))
        rectified_z = gt * np.einsum("i,ihw->hw", r1[2], rays)
        truth = cv.remap(
            rectified_z.astype(np.float32),
            *maps[0],
            cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        truth[~np.isfinite(truth) | (truth <= 0)] = np.nan
    return StereoResult(
        depth,
        np.where(valid, d, np.nan),
        valid,
        rect_left,
        rect_right,
        truth,
        baseline,
        float(p1[0, 0]),
    )


def evaluate_depth(estimated, ground_truth):
    """Error on common valid pixels, with coverage to expose sparse estimates."""
    a, b = np.asarray(estimated), np.asarray(ground_truth)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Depth maps must have equal 2D shapes")
    reference = np.isfinite(b) & (b > 0)
    valid = reference & np.isfinite(a) & (a > 0)
    error = a[valid] - b[valid]
    return dict(
        valid_pixels=int(valid.sum()),
        reference_pixels=int(reference.sum()),
        coverage=float(valid.sum() / max(1, reference.sum())),
        mae_m=float(np.abs(error).mean()) if error.size else None,
        rmse_m=float(np.sqrt(np.mean(error**2))) if error.size else None,
    )
