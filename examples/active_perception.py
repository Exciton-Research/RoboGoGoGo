"""Native wrist-camera exploration: a top view lacks the second grasp flank.

A fixed reference camera sees one flank; the free wrist must move sideways to
observe the other. Rendered RGB-D is an analytic test sensor, not hardware data.
"""

import json

import jax
import numpy as np
from camera_recovery import render_camera
from recovery_search import fixture

from robogogogo.perception import detect, fuse
from robogogogo.vla.cameras import world_camera_pose


def wrist_camera(session):
    pose = session.poses()
    body = session.arms[1].body
    r = np.asarray(pose.rotation[body])
    direction = np.array([0.2, -0.09, 0.5]) - np.asarray(pose.position[body])
    return dict(
        name="wrist_free",
        body=body,
        position=[0, 0, 0],
        look_at=(r.T @ direction).tolist(),
        up=(r.T @ np.array([0.0, 0.0, 1.0])).tolist(),
        vertical_fov_degrees=float(np.rad2deg(2 * np.arctan(240 / (2 * 220)))),
    )


def observe(session, m, camera):
    pose = session.poses()
    optical = world_camera_pose(
        camera, dict(positions=np.asarray(pose.position), rotations=np.asarray(pose.rotation))
    )
    position = np.asarray(optical["position"])
    target = position + np.asarray(optical["rotation"])[:, 2]
    boxes = [(np.asarray(m.state.position), [0.04, 0.30, 0.04], [170, 70, 140])]
    boxes += [
        ([cx, cy, top / 2], [2 * hx, 2 * hy, top], [80, 110, 80])
        for cx, cy, hx, hy, top in m.supports
    ]
    clouds = [
        render_camera(position, target, boxes, timestamp=session.elapsed),
        render_camera([0.85, 0.4, 0.8], [0.3, 0, 0.48], boxes, timestamp=session.elapsed),
    ]
    body = m.arms["left"].body
    seed = np.asarray(pose.position[body]) + 0.2 * np.asarray(pose.rotation[body])[:, 0]
    return detect(fuse(clouds), seed=seed)


def run():
    session, m, world = fixture()
    camera = wrist_camera(session)
    initial_count = None
    for i in range(5000):
        if m.owner and i > 250 and m.recovery is None:
            m.enable_recovery(camera_based=True, active_cameras=[camera])
            m.configure_gripper("left", mu_static=0.04, mu_dynamic=0.02)
            m.recovery.planner.observation = observe(session, m, camera)
            initial_count = len(m.recovery.planner.observation.grasps)
        if m.recovery:
            if m.recovery.attempts == 0:
                session.target_q = session.target_q.at[1].add(0.0003)
            if i % 10 == 0:
                m.recovery.planner.observation = observe(session, m, camera)
        session.advance(collision_world=world)
        if m.failure or (m.recovery and (m.recovery.secured or m.recovery.placed)):
            break
    return dict(initial_grasps=initial_count, time_s=session.elapsed, **m.summary())


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    result = run()
    print(
        json.dumps(
            {k: result[k] for k in ("initial_grasps", "time_s", "failure", "recovery")}, indent=2
        )
    )
