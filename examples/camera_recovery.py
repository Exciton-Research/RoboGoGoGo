"""Camera-only proposal demo using two synthetic calibrated RGB-D sensors.

The tiny ray renderer is a test sensor, separate from the detector. No object
geometry, body IDs or support coordinates are passed to the perception algorithm.
"""

import json

import numpy as np

from robogogogo.perception.recovery import backproject, detect, fuse


def render_camera(position, target, boxes, *, timestamp=0.0, width=320, height=240):
    position, target = np.asarray(position), np.asarray(target)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(forward, right), forward))
    k = np.array([[220.0, 0, width / 2], [0, 220.0, height / 2], [0, 0, 1.0]])
    yy, xx = np.indices((height, width))
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(k).T @ rotation.T
    depth = np.full((height, width), np.inf)
    rgb = np.zeros((height, width, 3), np.uint8)
    for center, size, color in boxes:
        low, high = (
            np.asarray(center) - np.asarray(size) / 2,
            np.asarray(center) + np.asarray(size) / 2,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (low - position) / rays, (high - position) / rays
        near, far = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        hit = (near > 0) & (far >= near) & (near < depth)
        depth[hit], rgb[hit] = near[hit], color
    depth[~np.isfinite(depth)] = np.nan
    return backproject(
        depth,
        rgb,
        k,
        position,
        rotation,
        timestamp=timestamp,
        uncertainty_m=0.002,
        source="synthetic_rgbd",
    )


def observe(offset=(0.0, 0.0, 0.0)):
    offset = np.asarray(offset)
    boxes = [
        (np.array([0.2, 0, 0.5]) + offset, [0.04, 0.30, 0.04], [170, 70, 140]),
        (np.array([0.48, 0, 0.425]) + offset, [0.25, 0.4, 0.05], [80, 110, 80]),
    ]
    clouds = [
        render_camera(np.array(position) + offset, np.array([0.3, 0, 0.48]) + offset, boxes)
        for position in [(-0.3, -0.4, 0.8), (0.85, 0.4, 0.8)]
    ]
    return fuse(clouds)


def physical_sensor(session, manipulation):
    """Analytic RGB-D test sensor; simulator state is used only to render test images.

    This box sensor approximates the specimen; it does not render deformed triangles.
    Perception receives pixels/calibration and a proprioceptive selection cue only.
    """
    boxes = [(np.asarray(manipulation.state.position), [0.04, 0.30, 0.04], [170, 70, 140])]
    boxes += [
        ([cx, cy, top / 2], [2 * hx, 2 * hy, top], [80, 110, 80])
        for cx, cy, hx, hy, top in manipulation.supports
    ]
    clouds = [
        render_camera(position, [0.3, 0, 0.48], boxes, timestamp=session.elapsed)
        for position in [(-0.3, -0.4, 0.8), (0.85, 0.4, 0.8)]
    ]
    holder = next(name for name, g in manipulation.grippers.items() if g["holding"])
    poses = session.poses()
    body = manipulation.arms[holder].body
    seed = np.asarray(poses.position[body]) + 0.2 * np.asarray(poses.rotation[body])[:, 0]
    return detect(fuse(clouds), seed=seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-helper", action="store_true")
    args = parser.parse_args()
    if args.physical_helper:
        import jax
        from recovery_search import run

        jax.config.update("jax_enable_x64", True)
        result = run("helper", camera_provider=physical_sensor)
        print(json.dumps({key: result[key] for key in ("time_s", "failure", "recovery")}, indent=2))
    else:
        result = detect(observe(), seed=[0.2, -0.09, 0.5])
        print(
            json.dumps(
                dict(**result.summary(), grasps=result.grasps, supports=result.supports), indent=2
            )
        )
