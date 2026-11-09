"""Bounded Cartesian candidate search for recovery, using the live payload envelope.

Paths preserve held-tool poses relative to one another. Every IK edge is checked;
this is finite candidate search, not a complete motion or grasp planner.
"""

import time
from copy import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import spatial as sp
from .control import inverse_kinematics_poses
from .planning.motion import PlanningError


@dataclass
class RecoveryPath:
    joints: list
    positions: list
    centers: list
    helper: str | None = None
    support: int | None = None
    contact_start: int | None = None
    grasp_position: list | None = None
    support_region: list | None = None


def contact_world(world, m, helper=None, support=None):
    """Only intended helper finger patches / selected support gain exemptions."""
    if world is None:
        return None
    from .planning.collision import BoxObstacle

    result = copy(world)
    result.obstacle_exclusions = set(world.obstacle_exclusions)
    result.support_contacts = set(world.support_contacts)
    if helper is not None:
        body = m.arms[helper].body
        fingers = {
            i for i, c in enumerate(world.capsules) if c.body == body and ":finger:" in c.name
        }
        payloads = {i for i, c in enumerate(world.capsules) if c.name == "payload"}
        result.pairs = np.asarray(
            [
                (i, j)
                for i, j in world.pairs
                if not ((i in fingers and j in payloads) or (j in fingers and i in payloads))
            ],
            dtype=int,
        ).reshape(-1, 2)
        result.obstacle_exclusions |= {(world.capsules[i].name, "transfer_bar") for i in fingers}
    if support is not None:
        target = m.supports[support]
        for obstacle in world.obstacles:
            if isinstance(obstacle, BoxObstacle) and np.allclose(
                [
                    *obstacle.center[:2],
                    obstacle.size[0] / 2,
                    obstacle.size[1] / 2,
                    obstacle.center[2] + obstacle.size[2] / 2,
                ],
                target,
            ):
                result.support_contacts.add(("payload", obstacle.name))
    return result


class RecoveryPlanner:
    def __init__(self, model, arms):
        self.model, self.arms = model, arms
        self.bodies = tuple(a.body for a in arms.values())
        self.solve = jax.jit(
            lambda q, p, r, mask: inverse_kinematics_poses(
                model, p, r, self.bodies, q, iterations=80, joint_mask=mask
            )
        )
        self.vision_required = False
        self.observation = None
        self.checked = 0
        self.rejections = []

    def observed(self, session):
        observation = self.observation
        if observation is None or not 0 <= session.elapsed - observation.timestamp <= 0.5:
            raise PlanningError(
                "Camera recovery requires a fresh calibrated observation (<= 0.5 s)"
            )
        return observation

    def support_candidates(self, m, session):
        if not self.vision_required:
            return [(i, s) for i, s in enumerate(m.supports)]
        observation = self.observed(session)
        candidates = []
        for proposal in observation.supports:
            position = np.asarray(proposal["position"])
            half = np.asarray(proposal["half_size"])
            # Associate only for physical support-load verification. Candidate position
            # and visible footprint remain camera-derived; never snap to table centers.
            for i, surface in enumerate(m.supports):
                if abs(position[2] - surface[4]) <= observation.uncertainty_m + 0.003 and np.all(
                    np.abs(position[:2] - surface[:2]) + half <= surface[2:4]
                ):
                    candidates.append((i, np.r_[position[:2], half, position[2]]))
                    break
        if not candidates:
            raise PlanningError("No observed landing patch with verifiable physical support")
        return candidates

    def require_world(self, world):
        if world is None or not (
            any(c.name == "payload" for c in world.capsules)
            and any(c.name != "payload" for c in world.capsules)
        ):
            raise PlanningError(
                "Recovery search requires robot and carried payload collision geometry"
            )

    def path(
        self,
        m,
        session,
        world,
        destinations,
        moving,
        *,
        helper=None,
        contact_segment=None,
        deadline=None,
    ):
        """Destinations are absolute tool positions; rotations stay at measured poses."""
        q = np.asarray(session.state.q)
        pose = session.poses()
        p = np.asarray(pose.position)[list(self.bodies)].copy()
        rotations = np.asarray(pose.rotation)[list(self.bodies)]
        initial_p = p.copy()
        center = np.asarray(m.state.position)
        segment = m.object_segment()
        carrier = next((i for i, n in enumerate(self.arms) if m.grippers[n]["holding"]), None)
        if carrier is None:
            raise PlanningError("No retaining gripper for recovery planning")
        fixed = {
            j for i, a in enumerate(self.arms.values()) if not moving[i] for j in a.joint_indices
        }
        movable = {
            j for i, a in enumerate(self.arms.values()) if moving[i] for j in a.joint_indices
        } - fixed
        mask = jnp.asarray([i in movable for i in range(len(q))])
        joints, positions, centers = [q.copy()], [p.copy()], [center.copy()]
        contact_start = None
        previous_bound = segment.copy()
        for index, destination in enumerate(destinations):
            destination = np.asarray(destination)
            start_p = p.copy()
            steps = max(
                1, int(np.ceil(np.max(np.linalg.norm(destination - start_p, axis=1)) / 0.015))
            )
            intended = helper if contact_segment is not None and index >= contact_segment else None
            oracle = contact_world(world, m, intended)
            # Retained grasps remain intentional contacts during bimanual transport,
            # including a helper recruited after the original task world was built.
            for name, grip in m.grippers.items():
                if grip["holding"]:
                    oracle = contact_world(oracle, m, name)
            if intended and contact_start is None:
                contact_start = len(joints)
            for step in range(1, steps + 1):
                if deadline is not None and time.monotonic() > deadline:
                    raise PlanningError("Recovery candidate search exceeded time budget")
                target = start_p + (destination - start_p) * step / steps
                candidate, errors, angles = self.solve(
                    jnp.asarray(q), jnp.asarray(target), jnp.asarray(rotations), mask
                )
                if np.max(errors) > 0.002 or np.max(angles) > 0.03:
                    raise PlanningError(
                        "Candidate pose is unreachable without moving retained arms"
                    )
                candidate = np.asarray(candidate)
                actual = np.asarray(
                    sp.forward_kinematics(self.model, jnp.asarray(candidate)).position
                )[list(self.bodies)]
                offset = actual[carrier] - initial_p[carrier]
                bound = segment.copy()
                bound[:2] += offset
                if len(bound) == 5:
                    bound[3:] += offset
                self.checked += 1
                if not oracle.edge_free(
                    q, candidate, object_start=previous_bound, object_end=bound
                ):
                    raise PlanningError("Candidate path violates robot/payload clearance")
                joints.append(candidate)
                positions.append(actual)
                centers.append(center + offset)
                q, p, previous_bound = candidate, actual, bound
        return RecoveryPath(joints, positions, centers, helper=helper, contact_start=contact_start)

    def helper_path(self, m, session, world, primary):
        self.require_world(world)
        pose = session.poses()
        p = np.asarray(pose.position)[list(self.bodies)]
        rotations = np.asarray(pose.rotation)[list(self.bodies)]
        origin = np.asarray(m.state.position)
        axis = np.asarray(m.state.rotation)[:, 1]
        if self.vision_required:
            observation = self.observed(session)
            axis = observation.object_axis
            if axis is None or not observation.grasps:
                raise PlanningError("No camera grasp with observed opposing surfaces")
            candidates = [np.asarray(g["position"]) for g in observation.grasps]
        elif m.structural:
            cells = np.asarray(m.contact.structure.position).reshape(-1, 8, 3).mean(1)
            candidates = [cells[i] for i in (len(cells) - 2, 1, len(cells) // 2)]
        else:
            candidates = [origin + offset * axis for offset in (0.09, -0.09, 0.0)]
        held_centers = [
            p[i] + 0.2 * rotations[i, :, 0]
            for i, n in enumerate(self.arms)
            if m.grippers[n]["holding"]
        ]
        deadline = time.monotonic() + 12
        for h, name in enumerate(self.arms):
            if name == primary or m.grippers[name]["holding"]:
                continue
            if m.grippers[name]["width"] < 0.09:
                self.rejections.append(dict(candidate=name, reason="Helper jaws are not open"))
                continue
            # Keep calibrated tool orientation; sample reachable sites along the object.
            if abs(np.dot(rotations[h, :, 1], axis)) > 0.35:
                self.rejections.append(
                    dict(candidate=name, reason="Closing axis is not transverse to specimen")
                )
                continue
            for candidate_index, point in enumerate(candidates):
                if self.vision_required:
                    proposal = observation.grasps[candidate_index]
                    if abs(np.dot(rotations[h, :, 1], proposal["closing_axis"])) < 0.95:
                        continue
                if any(np.linalg.norm(point - c) < 0.085 for c in held_centers):
                    continue
                goal = p.copy()
                goal[h] = point - 0.2 * rotations[h, :, 0]
                pre = goal.copy()
                pre[h] -= 0.08 * rotations[h, :, 0]
                # Lifted doglegs also allow approaches around a nearby obstacle.
                for rise in (0.0, 0.12):
                    waypoints = [pre, goal]
                    if rise:
                        elevated = p.copy()
                        elevated[h, 2] += rise
                        above = pre.copy()
                        above[h, 2] += rise
                        waypoints = [elevated, above, pre, goal]
                    try:
                        result = self.path(
                            m,
                            session,
                            world,
                            waypoints,
                            np.arange(len(p)) == h,
                            helper=name,
                            contact_segment=len(waypoints) - 1,
                            deadline=deadline,
                        )
                        result.grasp_position = point.tolist()
                        return result
                    except PlanningError as error:
                        self.rejections.append(
                            dict(candidate=name, position=point.tolist(), reason=str(error))
                        )
                        if time.monotonic() > deadline:
                            raise PlanningError(
                                "No verified helper path within search budget"
                            ) from error
        raise PlanningError("No reachable collision-free helper grasp candidate")

    def support_path(self, m, session, world):
        self.require_world(world)
        center = np.asarray(m.state.position)
        bound = m.object_segment()
        bottom = (
            bound[3, 2]
            if len(bound) == 5
            else center[2] - m.object.radius - m.object.half_length * abs(m.state.rotation[2, 1])
        )
        extent = center[2] - bottom
        p = np.asarray(session.poses().position)[list(self.bodies)].copy()
        moving = np.array([g["holding"] for g in m.grippers.values()])
        candidates = sorted(
            (
                (i, s)
                for i, s in self.support_candidates(m, session)
                if min(s[2:4]) > 0.015 and np.linalg.norm(center[:2] - s[:2]) <= 0.5
            ),
            key=lambda item: np.linalg.norm(center[:2] - item[1][:2]),
        )[: 8 if self.vision_required else 4]
        deadline = time.monotonic() + 15
        for index, support in candidates:
            goal_center = np.array([*support[:2], support[4] + extent + 0.05])
            for rise, side in ((0.06, 0.0), (0.14, 0.0), (0.10, 0.15), (0.10, -0.15)):
                travel_z = max(center[2], goal_center[2]) + rise
                centers = [np.array([*center[:2], travel_z])]
                if side:
                    delta = goal_center[:2] - center[:2]
                    perpendicular = np.array([-delta[1], delta[0]]) / max(
                        np.linalg.norm(delta), 1e-8
                    )
                    centers.append(
                        np.r_[(center[:2] + goal_center[:2]) / 2 + side * perpendicular, travel_z]
                    )
                centers += [np.array([*goal_center[:2], travel_z]), goal_center]
                destinations = [p + moving[:, None] * (c - center) for c in centers]
                try:
                    result = self.path(m, session, world, destinations, moving, deadline=deadline)
                    result.support = index
                    result.support_region = support.tolist()
                    return result
                except PlanningError as error:
                    self.rejections.append(dict(support=index, reason=str(error)))
                    if time.monotonic() > deadline:
                        raise PlanningError(
                            "No verified support path within search budget"
                        ) from error
        raise PlanningError("No nearby reachable support with a collision-free transport path")
