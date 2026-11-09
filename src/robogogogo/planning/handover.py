"""Pick, carry, guarded transfer, place: pose-aware planning with a carried payload."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .. import spatial as sp
from ..control import inverse_kinematics_poses
from .collision import (
    BoxObstacle,
    Capsule,
    CapsuleObstacle,
    CollisionWorld,
    adjacent_exclusions,
    capsules_from_visuals,
)
from .motion import MotionPlan, PlanningError, plan_motion
from .task import TaskPlan, TaskStage


@dataclass(frozen=True)
class HandoverPhase:
    carrier: str | None
    object_position: tuple[float, float, float]
    allowed_grasps: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()


class HandoverScene:
    def __init__(self, model, arms, visuals):
        if len(arms) != 2 or set(arms[0].joint_indices) & set(arms[1].joint_indices):
            raise ValueError("Handover requires two independent arms")
        self.model, self.arms = model, tuple(arms)
        # Use physical arm roots to establish the donor and receiver.
        positions = np.asarray(sp.forward_kinematics(model, model.home).position)
        ordered = sorted(arms, key=lambda a: positions[a.body, 1], reverse=True)
        self.giver, self.receiver = ordered
        self.pick = np.array([0.75, 0.28, 0.48])
        self.exchange = np.array([0.82, 0, 0.65])
        self.place = np.array([0.75, -0.28, 0.48])
        self.tables = tuple(
            BoxObstacle(
                name,
                (float(p[0]), float(p[1]), float((p[2] - 0.03) / 2)),
                (0.16, 0.08, float(p[2] - 0.03)),
            )
            for name, p in [("pickup_support", self.pick), ("place_support", self.place)]
        )
        self.rotations = {
            self.giver.name: np.asarray(sp.rpy_matrix(jnp.array([np.pi / 2, 0.0, 0.0]))),
            self.receiver.name: np.asarray(sp.rpy_matrix(jnp.array([np.pi / 2, 0.0, 0.0]))),
        }
        capsules = list(capsules_from_visuals(model, visuals))
        # Union of capsules encloses the full jaw sweep without inflating the
        # finger's perpendicular thickness into the nearby support.
        for arm in arms:
            capsules.append(
                Capsule(f"{arm.name}:palm", arm.body, (0.008, -0.06, 0), (0.008, 0.06, 0), 0.022)
            )
            for side in (-1, 1):
                for sample, gap in enumerate(np.linspace(0.037, 0.057, 5)):
                    capsules.append(
                        Capsule(
                            f"{arm.name}:finger:{side}:{sample}",
                            arm.body,
                            (0.02, side * gap, 0),
                            (0.23, side * gap, 0),
                            0.010,
                        )
                    )
        self.capsules = tuple(capsules)
        self.phases = []

    def world(self, phase, manipulation=None):
        capsules = list(self.capsules)
        obstacles = list(self.tables)
        exclusions = []
        named = []
        supports = []
        carrier = phase.carrier
        if carrier:
            arm = next(a for a in self.arms if a.name == carrier)
            # Nominal carried geometry is used for planning only. Runtime collision
            # checks override this segment with the independently integrated object.
            center = np.array([0.20, 0, 0.09 if carrier == self.giver.name else -0.09])
            axis = self.rotations[carrier].T @ np.array([0, 1, 0])
            payload = Capsule(
                "payload", arm.body, tuple(center - 0.12 * axis), tuple(center + 0.12 * axis), 0.03
            )
            capsules.append(payload)
            for allowed in phase.allowed_grasps:
                body = next(a.body for a in self.arms if a.name == allowed)
                for c in capsules:
                    if c.name != "payload" and c.body == body and ":finger:" in c.name:
                        named.append(("payload", c.name))
            supports = [("payload", name) for name in phase.supports]
        else:
            center = np.asarray(phase.object_position)
            axis = np.array([0, 1, 0])
            if manipulation is not None and manipulation.owner is None:
                center = np.asarray(manipulation.state.position)
                axis = np.asarray(manipulation.state.rotation)[:, 1]
            obstacles.append(
                CapsuleObstacle(
                    "transfer_bar", tuple(center - 0.12 * axis), tuple(center + 0.12 * axis), 0.03
                )
            )
            for allowed in phase.allowed_grasps:
                body = next(a.body for a in self.arms if a.name == allowed)
                exclusions += [
                    (c.name, "transfer_bar")
                    for c in capsules
                    if c.body == body and ":finger:" in c.name
                ]
        return CollisionWorld(
            self.model,
            capsules,
            obstacles=obstacles,
            margin=0.015,
            excluded_pairs=adjacent_exclusions(self.model),
            excluded_capsule_pairs=named,
            obstacle_exclusions=exclusions,
            support_contacts=supports,
        )

    def plan(self, initial_q, *, seed=7):
        left, right = self.giver.name, self.receiver.name
        park = {left: [0.62, 0.45, 0.70], right: [0.62, -0.45, 0.70]}
        pick_tool = (self.pick + [-0.20, 0.09, 0]).tolist()
        transfer_left = (self.exchange + [-0.20, 0.09, 0]).tolist()
        transfer_right = (self.exchange + [-0.20, -0.09, 0]).tolist()
        place_tool = (self.place + [-0.20, -0.09, 0]).tolist()
        specifications = [
            (
                "Open and stage grippers",
                None,
                park,
                None,
                HandoverPhase(None, tuple(self.pick)),
            ),
            (
                "Approach pickup",
                left,
                {**park, left: pick_tool},
                None,
                HandoverPhase(None, tuple(self.pick), (left,)),
            ),
            (
                "Grasp object",
                left,
                {**park, left: pick_tool},
                "grasp",
                HandoverPhase(None, tuple(self.pick), (left,)),
            ),
            (
                "Lift object",
                left,
                {**park, left: (np.array(pick_tool) + [0, 0, 0.12]).tolist()},
                None,
                HandoverPhase(left, tuple(self.pick), (), ("pickup_support",)),
            ),
            (
                "Present object",
                left,
                {**park, left: transfer_left},
                None,
                HandoverPhase(left, tuple(self.exchange)),
            ),
            (
                "Align receiver",
                left,
                {left: transfer_left, right: transfer_right},
                None,
                HandoverPhase(left, tuple(self.exchange), (right,)),
            ),
            (
                "Secure receiver and release giver",
                right,
                {left: transfer_left, right: transfer_right},
                "transfer",
                HandoverPhase(left, tuple(self.exchange), (right,)),
            ),
            (
                "Giver retreats",
                right,
                {**park, right: transfer_right},
                None,
                HandoverPhase(right, tuple(self.exchange), (left,)),
            ),
            (
                "Carry to destination",
                right,
                {**park, right: (np.array(place_tool) + [0, 0, 0.12]).tolist()},
                None,
                HandoverPhase(right, tuple(self.place)),
            ),
            (
                "Lower onto support",
                right,
                {**park, right: place_tool},
                None,
                HandoverPhase(right, tuple(self.place), (), ("place_support",)),
            ),
            (
                "Release object",
                right,
                {**park, right: place_tool},
                "release",
                HandoverPhase(right, tuple(self.place), (), ("place_support",)),
            ),
            (
                "Clear placed object",
                None,
                park,
                "verify_placement",
                HandoverPhase(None, tuple(self.place), (right,)),
            ),
        ]
        # Contact-aware insertion/withdrawal: open jaws enter the grasp along
        # their approach axis, rather than sweeping through the object on an RRT path.
        expanded = []
        for name, owner, targets, operation, phase in specifications:
            if name in ("Approach pickup", "Align receiver"):
                active = left if name == "Approach pickup" else right
                pre = {k: list(v) for k, v in targets.items()}
                pre[active][0] -= 0.12
                expanded.append(
                    (
                        "Pregrasp giver" if active == left else "Pregrasp receiver",
                        owner,
                        pre,
                        None,
                        replace(phase, allowed_grasps=()),
                    )
                )
            if name == "Giver retreats":
                withdraw = {
                    left: (np.array(transfer_left) - [0.12, 0, 0]).tolist(),
                    right: transfer_right,
                }
                expanded.append(("Withdraw giver", right, withdraw, None, phase))
            if name == "Clear placed object":
                withdraw = {**park, right: (np.array(place_tool) - [0.12, 0, 0]).tolist()}
                expanded.append(("Withdraw receiver", None, withdraw, None, phase))
            if name in ("Giver retreats", "Clear placed object"):
                phase = replace(phase, allowed_grasps=())
            expanded.append((name, owner, targets, operation, phase))
        specifications = expanded
        bodies = tuple(a.body for a in self.arms)
        rotations = jnp.asarray([self.rotations[a.name] for a in self.arms])
        solve = jax.jit(lambda q, p: inverse_kinematics_poses(self.model, p, rotations, bodies, q))
        current = np.asarray(initial_q, dtype=float)
        stages = []
        worlds = []
        for index, (name, owner, targets, operation, phase) in enumerate(specifications):
            world = self.world(phase)
            goals = jnp.asarray([targets[a.name] for a in self.arms])
            rng = np.random.default_rng(seed + index)
            candidates = [current, np.asarray(self.model.home).copy()]
            preferred = np.asarray(self.model.home).copy()
            for arm in self.arms:
                preferred[list(arm.joint_indices)[3]] = np.pi / 2
            candidates.insert(1, preferred)
            for attempt in range(48):
                candidate = current.copy()
                selected = self.arms[attempt % len(self.arms)]
                candidate[list(selected.joint_indices)] = rng.uniform(
                    -2.6, 2.6, len(selected.joint_indices)
                )
                candidates.append(candidate)
            goal = None
            failure = "IK tolerance"
            for candidate in candidates:
                solved, error, angles = solve(jnp.asarray(candidate), goals)
                if np.max(error) < 0.004 and np.max(angles) < 0.035:
                    failure = str(world.clearance(solved))
                if (
                    np.isfinite(solved).all()
                    and np.max(error) < 0.004
                    and np.max(angles) < 0.035
                    and world.clearance(solved).distance >= world.margin
                ):
                    goal = solved
                    break
            if goal is None:
                raise PlanningError(
                    f"{name}: no collision-free pose IK solution within the search budget: {failure}"
                )
            try:
                if name in (
                    "Approach pickup",
                    "Lift object",
                    "Align receiver",
                    "Withdraw giver",
                    "Lower onto support",
                    "Withdraw receiver",
                ):
                    # Keep the bar level near supports. Endpoint-only pose IK can
                    # rotate a wrist substantially between two level endpoints.
                    beginning = np.array([stages[-1].targets[a.name] for a in self.arms])
                    path = [tuple(current)]
                    for fraction in np.linspace(0.0, 1.0, 9)[1:]:
                        desired = beginning + fraction * (np.asarray(goals) - beginning)
                        candidate, error, angle = solve(jnp.asarray(path[-1]), jnp.asarray(desired))
                        if (
                            np.max(error) > 0.004
                            or np.max(angle) > 0.035
                            or not world.edge_free(path[-1], candidate)
                        ):
                            raise PlanningError(
                                "No level Cartesian approach within collision margins"
                            )
                        path.append(tuple(np.asarray(candidate)))
                    goal = candidate
                    motion = MotionPlan(tuple(path), "cartesian", world.checks)
                else:
                    motion = plan_motion(world, current, np.asarray(goal), seed=seed + index)
            except PlanningError as error:
                raise PlanningError(f"{name}: {error}") from error
            stages.append(
                TaskStage(
                    name,
                    owner,
                    targets,
                    motion.waypoints,
                    motion.method,
                    0.2,
                    operation,
                    {k: v.tolist() for k, v in self.rotations.items()},
                )
            )
            self.phases.append(phase)
            worlds.append(world)
            current = np.asarray(goal)
        return TaskPlan(
            "Pick, hand over, and place", tuple(stages), seed, tuple(self.exchange), "handover"
        ), worlds
