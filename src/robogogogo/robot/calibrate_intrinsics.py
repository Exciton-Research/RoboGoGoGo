"""Measured pinhole calibration from checkerboard images with held-out validation.

python -m robogogogo.hardware.calibrate_intrinsics --help
This estimates intrinsics only, never camera mounts or clock offsets.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robogogogo.perception.stereo import opencv


def fit_corners(corners, image_size, pattern, square_m):
    cv = opencv()
    if len(corners) < 14 or not 0 < square_m < 1 or min(pattern) < 3:
        raise ValueError(
            "Provide at least 14 varied checkerboard views, >=3 inner corners per axis, and square size in meters"
        )
    views = [np.asarray(points, np.float32).reshape(-1, 1, 2) for points in corners]
    count = pattern[0] * pattern[1]
    if any(len(points) != count or not np.isfinite(points).all() for points in views):
        raise ValueError("Each image must contain the full detected board")
    centers = np.array([points[:, 0].mean(axis=0) for points in views])
    sizes = np.array([np.linalg.norm(np.ptp(points[:, 0], axis=0)) for points in views])
    if (np.ptp(centers, axis=0) / image_size < 0.1).any() or sizes.max() / max(
        sizes.min(), 1e-6
    ) < 1.2:
        raise ValueError("Move the board across both image axes and vary its distance")
    objects = np.zeros((count, 3), np.float32)
    objects[:, :2] = np.mgrid[: pattern[0], : pattern[1]].T.reshape(-1, 2) * square_m
    held_out = [i for i in range(len(views)) if i % 4 == 0]
    training = [i for i in range(len(views)) if i not in held_out]
    rms, k, d, rotations, _ = cv.calibrateCamera(
        [objects] * len(training), [views[i] for i in training], image_size, None, None
    )
    normals = np.array([cv.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    if max(np.linalg.norm(normal - normals[0]) for normal in normals) < 0.15:
        raise ValueError("Calibration needs tilted board views, not just front-facing translations")
    errors = []
    for points in views:
        solved, rotation, translation = cv.solvePnP(objects, points, k, d)
        if not solved:
            raise ValueError("Could not estimate a validation board pose")
        projected, _ = cv.projectPoints(objects, rotation, translation, k, d)
        errors.append(float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=2)))))
    validation = max(errors[i] for i in held_out)
    if (
        not np.isfinite(k).all()
        or not np.isfinite(d).all()
        or not np.isfinite(errors).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or rms > 1
        or validation > 1
        or max(errors) > 1.5
    ):
        raise ValueError(
            "Calibration failed reprojection acceptance: train <=1 px, held-out <=1 px, all <=1.5 px"
        )
    return dict(
        width=image_size[0],
        height=image_size[1],
        intrinsics=k.tolist(),
        distortion=d.ravel().tolist(),
        distortion_model="plumb_bob",
        reprojection_error_px=max(float(rms), validation),
        training_rms_px=float(rms),
        held_out_max_rms_px=validation,
        per_image_rms_px=errors,
        held_out_indices=held_out,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        pattern_inner_corners=list(pattern),
        square_size_m=square_m,
    )


def calibrate_images(paths, pattern, square_m):
    cv = opencv()
    corners, hashes, size = [], [], None
    for path in paths:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate calibration image; capture distinct board poses")
        image = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable calibration image: {path}")
        current = (image.shape[1], image.shape[0])
        if size is not None and current != size:
            raise ValueError("All calibration images must use the deployment resolution")
        size = current
        found, points = cv.findChessboardCornersSB(
            image, pattern, flags=cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            raise ValueError(f"Full checkerboard not detected: {path}")
        corners.append(points)
        hashes.append(digest)
    report = fit_corners(corners, size, pattern, square_m)
    report["image_sha256"] = hashes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--columns", type=int, required=True, help="Inner corners across the board")
    parser.add_argument("--rows", type=int, required=True, help="Inner corners down the board")
    parser.add_argument("--square-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = calibrate_images(args.images, (args.columns, args.rows), args.square_m)
        with args.output.open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        f"Saved measured intrinsics to {args.output}; held-out RMS <= {report['held_out_max_rms_px']:.3f} px"
    )


if __name__ == "__main__":
    main()
