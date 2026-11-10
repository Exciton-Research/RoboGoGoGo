"""Camera calibration in robot/world frames without a rendering dependency."""

import numpy as np


def world_camera_pose(camera: dict, state: dict) -> dict:
    """Return optical camera-to-world extrinsics (+X right, +Y down, +Z forward)."""
    position = np.asarray(camera["position"], dtype=float)
    if camera.get("rpy") is not None:
        forward, up = mounting_vectors(camera["rpy"])
        target = position + forward
    else:
        target = np.asarray(camera["look_at"], dtype=float)
        up = np.asarray(camera["up"], dtype=float)
    body = camera.get("body")
    if body is not None:
        rotation = np.asarray(state["rotations"][body])
        translation = np.asarray(state["positions"][body])
        position = rotation @ position + translation
        target = rotation @ target + translation
        up = rotation @ up
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return dict(
        position=position.tolist(), rotation=np.column_stack((right, down, forward)).tolist()
    )


def mounting_vectors(rpy):
    """Mount-relative Rz(yaw) Ry(pitch) Rx(roll), +X forward, +Y left, +Z up."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return rotation[:, 0], rotation[:, 2]


def mounted_rig(model, arms, *, wrists=True):
    """Illustrative 64 mm head stereo pair; replace offsets with measured calibration."""
    if "head" not in model.names:
        raise ValueError(
            "This robot has no head link. Configure the actual camera mounts explicitly."
        )
    cameras = [
        dict(
            name=f"head_{side}",
            mount="head",
            position=[0.105, y, 0],
            rpy=[0, float(np.deg2rad(40)), 0],
            vertical_fov_degrees=70,
        )
        for side, y in [("left", 0.032), ("right", -0.032)]
    ]
    if wrists:
        cameras += [
            dict(
                name=f"wrist_{i + 1}",
                mount=arm.tool,
                position=[0, 0, 0.08],
                rpy=[0, float(np.deg2rad(20)), 0],
                vertical_fov_degrees=80,
            )
            for i, arm in enumerate(arms)
        ]
    if len(cameras) > 8:
        raise ValueError("Head + wrist rig exceeds the eight-camera limit; configure a smaller rig")
    return cameras
