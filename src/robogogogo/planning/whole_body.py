"""Position-only bimanual reaching with a tire-driven planar base.

Plans cover self-collision proxies and sampled static support, not environment
obstacles, tool orientation, carried loads or hardware safety certification.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from .. import spatial as sp
from ..robot.mobile_design import mobile_home
from .mobile import MobilePlanningWorld
from .motion import MotionPlan, PlanningError, plan_motion


class WholeBodyPlanner:
    def __init__(self, dynamics, arms):
        self.dynamics, self.model = dynamics, dynamics.model
        self.arms = tuple(arms)
        self.indices = np.array(
            sorted(
                set(dynamics.base.tolist())
                | {
                    self.model.indices[i]
                    for i, name in enumerate(self.model.names)
                    if self.model.indices[i] >= 0
                    and (
                        name.startswith(("left_", "right_"))
                        or name in ("torso_yaw", "torso_lower", "torso_upper", "torso")
                    )
                }
            )
        )
        self.positions = jax.jit(
            lambda q: sp.forward_kinematics(self.model, q).position[
                jnp.array([arm.body for arm in self.arms])
            ]
        )
        self.solve = jax.jit(self._solve)

    def _solve(self, start, seed, targets):
        weights = jnp.zeros_like(start).at[self.indices].set(1.0)
        weights = weights.at[self.dynamics.base[:2]].set(3.0)
        lower, upper = self.model.lower, self.model.upper
        base = self.dynamics.base
        lower = lower.at[base].set(
            jnp.maximum(lower[base], start[base] - jnp.array([0.7, 0.7, 0.5]))
        )
        upper = upper.at[base].set(
            jnp.minimum(upper[base], start[base] + jnp.array([0.7, 0.7, 0.5]))
        )

        def iteration(q, _):
            error = (targets - self.positions(q)).reshape(-1)
            jac = jax.jacfwd(self.positions)(q).reshape((-1, q.size))
            delta = weights * (
                jac.T
                @ jnp.linalg.solve((jac * weights) @ jac.T + 0.002 * jnp.eye(error.size), error)
            )
            step = jnp.full_like(q, 0.06).at[base[:2]].set(0.025)
            return jnp.clip(q + jnp.clip(delta, -step, step), lower, upper), None

        return jax.lax.scan(iteration, seed, None, length=120)[0]

    def world(self, start, deadline):
        world = MobilePlanningWorld(self.dynamics, start, self.indices, deadline)
        for axis, index in enumerate(self.dynamics.base.tolist()):
            j = list(self.indices).index(index)
            radius = 0.7 if axis < 2 else 0.5
            world.lower[j] = max(world.lower[j], start[index] - radius)
            world.upper[j] = min(world.upper[j], start[index] + radius)
        return world

    def plan(self, start, center, *, budget_seconds=15):
        start, center = np.asarray(start), np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise PlanningError("Whole-body target must be three finite world coordinates")
        positions = np.asarray(self.positions(start))
        displacement = center - positions.mean(axis=0)
        if np.linalg.norm(displacement) > 1.0:
            raise PlanningError("Whole-body reach is limited to a local one-meter target change")
        targets = positions + displacement
        # Compile the local solver before starting the bounded geometric search.
        solved = np.asarray(self.solve(start, start, targets))
        deadline = time.monotonic() + budget_seconds
        world = self.world(start, deadline)
        for posture in (None, "reach", "low"):
            if posture is not None:
                seed = start.copy()
                home = mobile_home(self.model, posture)
                for name in ("torso_lower", "torso_upper", "torso"):
                    index = self.model.indices[self.model.names.index(name)]
                    seed[index] = home[index]
                solved = np.asarray(self.solve(start, seed, targets))
            if np.max(np.linalg.norm(np.asarray(self.positions(solved)) - targets, axis=1)) > 0.008:
                continue
            a, b = start[self.indices], solved[self.indices]
            if world.clearance(b).distance < world.margin:
                continue
            if world.edge_free(a, b):
                plan = MotionPlan((tuple(start), tuple(solved)), "whole_body_direct", world.checks)
            else:
                reduced = plan_motion(world, a, b, seed=0, max_iterations=1500)
                plan = MotionPlan(
                    tuple(tuple(world.full(q)) for q in reduced.waypoints),
                    "whole_body_detour",
                    world.checks,
                )
            return plan, targets
        raise PlanningError(
            "No reachable, collision-clear whole-body configuration found; target unchanged"
        )
