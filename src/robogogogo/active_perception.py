"""Bounded, collision-checked camera exploration using actual articulated joints.

Views maximize predicted measurement gain using observed surface occlusion. Success
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
        if not 1 <= len(self.cameras) <= 8:
            raise ValueError("Active perception requires one to eight calibrated cameras")
        names = [c.get("name") for c in self.cameras]
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("Active camera names must be nonempty and unique")
        for camera in self.cameras:
            if camera.get("mount") is not None and camera["mount"] not in model.names:
                raise ValueError("Unknown active camera mount")
            try:
                pose = self.pose(camera, model.home)
                if (
                    not np.isfinite(pose["position"]).all()
                    or not np.isfinite(pose["rotation"]).all()
                    or not 1 < camera.get("vertical_fov_degrees", 70) < 179
                ):
                    raise ValueError("Invalid active camera calibration")
            except (KeyError, TypeError, IndexError) as error:
                raise ValueError("Invalid active camera calibration") from error
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
        self.selection = None
        self.ranking = []

    def summary(self):
        return dict(
            attempts=self.attempts,
            max_views=self.max_views,
            camera=self.selected_camera,
            capture_after=self.capture_after,
            selection=self.selection,
            ranking=self.ranking[:8],
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
        observation = m.recovery.planner.observation
        evidence = observation.view_evidence if observation is not None else None
        if evidence is None or not len(evidence.points):
            raise PlanningError("Active perception needs measured surface evidence")
        if not 0 <= session.elapsed - observation.timestamp <= 0.5:
            raise PlanningError("Active perception needs fresh surface evidence")
        self.ranking = []
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
            # Single-axis and coordinated two-axis proposals at two scales.
            # This bounded local search optimizes measured utility, not motion novelty.
            axes = []
            for joint in sorted(movable):
                link = self.model.indices.index(joint)
                step = 0.06 if self.model.kinds[link] == "prismatic" else 0.3
                direction = np.zeros_like(q)
                direction[joint] = step
                axes.append(direction)
            directions = [sign * axis for axis in axes for sign in (-1, 1)]
            directions += [
                a + sign * b for i, a in enumerate(axes) for b in axes[i + 1 :] for sign in (-1, 1)
            ]
            directions += [-d for d in directions[len(axes) * 2 :]]
            for direction in directions:
                for scale in (0.5, 1.0):
                    target = np.clip(q + scale * direction, self.model.lower, self.model.upper)
                    if np.linalg.norm(target - q) < 0.015:
                        continue
                    if any(np.linalg.norm(target - old) < 0.015 for old in self.visited):
                        continue
                    after = self.pose(camera, target)
                    quality = evidence.score(
                        before, after, camera, support=resume == "searching_support"
                    )
                    cost = 0.002 * np.linalg.norm(scale * direction)
                    score = quality["gain"] - cost
                    report = dict(camera=camera["name"], score=float(score), **quality)
                    if score <= 1e-6:
                        continue
                    candidates.append((score, camera["name"], target, report))
        self.ranking = [c[3] for c in sorted(candidates, key=lambda c: -c[0])]
        for _, name, target, report in sorted(candidates, key=lambda c: -c[0]):
            if world.edge_free(q, target, object_start=bound, object_end=bound):
                self.goal, self.selected_camera, self.resume = target, name, resume
                self.selection = report
                self.command = q.copy()
                self.reference_object = np.asarray(m.state.position).copy()
                self.attempts += 1
                self.visited.extend([q.copy(), target.copy()])
                self.dwell, self.capture_after = 0.0, None
                return
            self.rejections.append(
                dict(camera=name, reason="View motion violates collision clearance")
            )
        raise PlanningError(
            "No reachable camera view with positive information gain without moving a retaining grasp"
        )

    def advance(self, session, dt):
        q = np.asarray(session.state.q)
        rates = np.full(q.shape, 0.15)
        for kind, index in zip(self.model.kinds, self.model.indices):
            if index >= 0 and kind == "prismatic":
                rates[index] = 0.03
        self.command += np.clip(self.goal - self.command, -rates * dt, rates * dt)
        target = self.command.copy()
        quiet = (
            np.max(np.abs(q - self.goal)) < 0.002
            and np.max(np.abs(np.asarray(session.state.qd))) < 0.025
        )
        self.dwell = self.dwell + dt if quiet else 0.0
        return target, self.dwell >= 0.15
