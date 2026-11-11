"""Bounded self-collision planning for the single RG Mobile Duo embodiment."""

import time

import numpy as np

from ..robot.mobile_design import mobile_home
from .collision import Clearance
from .motion import MotionPlan, PlanningError, plan_motion


class MobilePlanningWorld:
    """Freeze wheels/base/head; search only this robot's arm and torso joints."""

    def __init__(self, dynamics, start, indices, deadline):
        self.dynamics, self.collision = dynamics, dynamics.collision_world
        self.start = np.asarray(start).copy()
        self.indices = np.asarray(indices, dtype=int)
        self.lower = np.asarray(dynamics.model.lower)[self.indices]
        self.upper = np.asarray(dynamics.model.upper)[self.indices]
        self.margin = self.collision.margin
        self.deadline = deadline
        self.checks = 0

    def full(self, q):
        result = self.start.copy()
        result[self.indices] = q
        return result

    def clearance(self, q):
        if time.monotonic() > self.deadline:
            raise PlanningError("Mobile planning budget exhausted; no goal was applied")
        self.checks += 1
        full = self.full(q)
        clearance = self.collision.clearance(full)
        if float(self.dynamics.static_margin(full)) < self.dynamics.margin:
            return Clearance(-1.0, ("support_margin", "robot"))
        return clearance

    def edge_free(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if min(self.clearance(a).distance, self.clearance(b).distance) < self.margin:
            return False
        try:
            if not self.collision.edge_free(self.full(a), self.full(b), deadline=self.deadline):
                return False
        except TimeoutError as error:
            raise PlanningError(str(error)) from error
        # Static support is sampled separately; execution retains dynamic ZMP
        # and per-substep collision checks under actual coupled dynamics.
        count = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.04)) + 1)
        return all(
            self.clearance(a + t * (b - a)).distance >= self.margin
            for t in np.linspace(0, 1, count)
        )


def plan_mobile(dynamics, start, goal, *, seed=0, budget_seconds=12):
    start, goal = np.asarray(start), np.asarray(goal)
    if (
        start.shape != dynamics.model.home.shape
        or goal.shape != start.shape
        or not np.isfinite(start).all()
        or not np.isfinite(goal).all()
    ):
        raise PlanningError("Mobile plan requires finite full-robot configurations")
    if not np.allclose(start[dynamics.driven], goal[dynamics.driven], atol=1e-6):
        raise PlanningError("Use mobile drive commands for base motion; arm plans freeze the base")
    indices = [
        dynamics.model.indices[i]
        for i, name in enumerate(dynamics.model.names)
        if dynamics.model.indices[i] >= 0
        and (
            name.startswith(("left_", "right_"))
            or name in ("torso_yaw", "torso_lower", "torso_upper", "torso")
        )
    ]
    if not np.allclose(np.delete(start, indices), np.delete(goal, indices), atol=1e-6):
        raise PlanningError("Arm/torso planning does not change head or base joints")
    world = MobilePlanningWorld(dynamics, start, indices, time.monotonic() + budget_seconds)
    a, b = start[indices], goal[indices]
    for label, q in (("start", a), ("goal", b)):
        clearance = world.clearance(q)
        if clearance.distance < world.margin:
            raise PlanningError(f"Self-collision/support {label} is invalid: {clearance.pair}")
    if world.edge_free(a, b):
        return MotionPlan((tuple(start), tuple(goal)), "direct", world.checks)
    # Deterministic retreat through a known neutral pose often avoids expensive
    # sampling; it is accepted only if both complete edges pass the oracle.
    neutral = mobile_home(dynamics.model)[indices]
    if world.edge_free(a, neutral) and world.edge_free(neutral, b):
        return MotionPlan(
            (tuple(start), tuple(world.full(neutral)), tuple(goal)), "neutral_detour", world.checks
        )
    reduced = plan_motion(world, a, b, seed=seed, max_iterations=2500, step_size=0.35)
    return MotionPlan(
        tuple(tuple(world.full(q)) for q in reduced.waypoints), reduced.method, world.checks
    )
