"""Bounded, collision-checked camera exploration using actual articulated joints.

View scores are geometric heuristics, not predicted occlusion guarantees. Success
requires a new measured observation after the camera has settled.
"""

import numpy as np

from . import spatial as sp
from .planning.motion import PlanningError
from .vla.cameras import world_camera_pose


class ActivePerception:
    def __init__(self, model, cameras, *, max_views=4):
        if not 1 <= max_views <= 8:
            raise ValueError("Active perception requires 1–8 bounded view attempts")
        self.model = model
        self.cameras = [dict(c) for c in cameras]
        self.max_views = max_views
        self.attempts = 0
        self.selected_camera = None
        self.goal = None
        self.capture_after = None
        self.previous_observation = None
        self.dwell = 0.0
        self.visited = []
        self.rejections = []
        self.reference_object = None
        self.resume = None

    def summary(self):
        return dict(
            attempts=self.attempts,
            max_views=self.max_views,
            camera=self.selected_camera,
            capture_after=self.capture_after,
            rejections=self.rejections[-8:],
        )

    def camera_body(self, camera):
        mount = camera.get("mount")
        body = self.model.names.index(mount) if mount in self.model.names else camera.get("body")
        if body is None or not 0 <= body < len(self.model.names):
            return None
        return body

    def ancestry(self, body):
        indices = set()
        while body >= 0:
            if self.model.indices[body] >= 0:
                indices.add(self.model.indices[body])
            body = self.model.parents[body]
        return indices

    def pose(self, camera, q):
        poses = sp.forward_kinematics(self.model, q)
        config = dict(camera, body=self.camera_body(camera))
        return world_camera_pose(
            config, dict(positions=np.asarray(poses.position), rotations=np.asarray(poses.rotation))
        )

    def begin(self, m, session, world, resume):
        if self.attempts >= self.max_views:
            raise PlanningError("Active perception exhausted its view budget")
        m.recovery.planner.require_world(world)
        holders = [n for n, g in m.grippers.items() if g["holding"]]
        if not holders:
            raise PlanningError("Active perception requires a retaining grasp")
        protected = {
            j
            for n, a in m.arms.items()
            if m.grippers[n]["holding"] or m.grippers[n]["command"] == "closed"
            for j in a.joint_indices
        }
        pose = session.poses()
        holder = m.arms[holders[0]].body
        cue = np.asarray(pose.position[holder]) + 0.2 * np.asarray(pose.rotation[holder])[:, 0]
        if resume == "searching_support":
            cue = cue + [0.1, 0.0, -0.15]
        q = np.asarray(session.state.q)
        bound = m.object_segment()
        candidates = []
        for camera in self.cameras:
            body = self.camera_body(camera)
            if body is None:
                continue
            movable = self.ancestry(body) - protected
            if not movable:
                self.rejections.append(
                    dict(camera=camera["name"], reason="Fixed mount or retaining-arm joints")
                )
                continue
            if not any(c.body == body for c in world.capsules):
                self.rejections.append(
                    dict(camera=camera["name"], reason="Camera body lacks collision geometry")
                )
                continue
            before = self.pose(camera, q)
            for joint in sorted(movable):
                link = self.model.indices.index(joint)
                step = 0.06 if self.model.kinds[link] == "prismatic" else 0.3
                for delta in (-step, step):
                    target = q.copy()
                    target[joint] = np.clip(
                        q[joint] + delta, self.model.lower[joint], self.model.upper[joint]
                    )
                    if abs(target[joint] - q[joint]) < step / 2:
                        continue
                    if any(np.linalg.norm(target - old) < 0.015 for old in self.visited):
                        continue
                    after = self.pose(camera, target)
                    center, rotation = np.asarray(after["position"]), np.asarray(after["rotation"])
                    ray = cue - center
                    distance = np.linalg.norm(ray)
                    if distance < 0.08:
                        continue
                    alignment = np.dot(rotation[:, 2], ray / distance)
                    # Keep the cue within the vertical viewing cone, allowing a small search margin.
                    if alignment < np.cos(
                        np.deg2rad(camera.get("vertical_fov_degrees", 70) / 2 + 10)
                    ):
                        continue
                    novelty = np.linalg.norm(center - before["position"]) + 0.15 * np.linalg.norm(
                        rotation - before["rotation"]
                    )
                    candidates.append(
                        (alignment + 2 * novelty - 0.1 * distance, camera["name"], target)
                    )
        for _, name, target in sorted(candidates, key=lambda c: -c[0]):
            if world.edge_free(q, target, object_start=bound, object_end=bound):
                self.goal, self.selected_camera, self.resume = target, name, resume
                self.reference_object = np.asarray(m.state.position).copy()
                self.attempts += 1
                self.visited.extend([q.copy(), target.copy()])
                self.dwell, self.capture_after = 0.0, None
                return
            self.rejections.append(
                dict(camera=name, reason="View motion violates collision clearance")
            )
        raise PlanningError("No reachable camera view without moving a retaining grasp")

    def advance(self, session, dt):
        q = np.asarray(session.state.q)
        rates = np.full(q.shape, 0.15)
        for kind, index in zip(self.model.kinds, self.model.indices):
            if index >= 0 and kind == "prismatic":
                rates[index] = 0.03
        target = q + np.clip(self.goal - q, -rates * dt, rates * dt)
        quiet = (
            np.max(np.abs(q - self.goal)) < 0.002
            and np.max(np.abs(np.asarray(session.state.qd))) < 0.025
        )
        self.dwell = self.dwell + dt if quiet else 0.0
        return target, self.dwell >= 0.15
