"""Deterministic policies implementing the same contract as learned models.

These baselines use privileged joint/target state, not visual understanding.
They do not provide collision avoidance or a hardware safety certification.
"""

import jax
import jax.numpy as jnp
import numpy as np

from robogogogo.contracts import ActionChunk, Observation
from robogogogo.control import inverse_kinematics_targets


class ReachPolicy:
    """Simultaneous multi-arm position IK with bounded joint increments."""

    def __init__(self, model, arms, *, control_ticks=10):
        self.model, self.arms = model, tuple(arms)
        if not self.arms or not 1 <= control_ticks <= 20:
            raise ValueError("Require arms and control_ticks in [1, 20]")
        self.control_ticks = control_ticks
        bodies = tuple(arm.body for arm in self.arms)
        mask = (
            jnp.zeros_like(model.home)
            .at[jnp.array(sorted({j for a in self.arms for j in a.joint_indices}))]
            .set(1)
        )
        self._solve = jax.jit(
            lambda q, targets: inverse_kinematics_targets(
                model, targets, bodies, q, joint_mask=mask
            )[0]
        )
        self.limit = np.array(
            [0.02 if k == "prismatic" else 0.12 for k in model.kinds if k != "fixed"]
        )

    def predict(self, observation: Observation) -> ActionChunk:
        q = np.asarray(observation.joint_position, dtype=float)
        targets = np.asarray([observation.targets[a.name] for a in self.arms])
        if q.shape != self.model.home.shape or not np.isfinite(q).all():
            raise ValueError("Invalid joint observation")
        if targets.shape != (len(self.arms), 3) or not np.isfinite(targets).all():
            raise ValueError("Invalid reach targets")
        goal = np.asarray(self._solve(jnp.asarray(q), jnp.asarray(targets)))
        return ActionChunk((tuple(np.clip(goal - q, -self.limit, self.limit)),), self.control_ticks)


class JointWaypointPolicy:
    """State machine that advances only after reaching and settling at each waypoint.

    A fresh instance or reset() starts a new episode. Finished policies hold position.
    """

    def __init__(self, waypoints, *, max_delta, tolerance=0.01, velocity_tolerance=0.05):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.limit = np.asarray(max_delta, dtype=float)
        if (
            self.waypoints.ndim != 2
            or not len(self.waypoints)
            or self.limit.shape != self.waypoints.shape[1:]
            or not np.isfinite(self.waypoints).all()
            or not np.isfinite(self.limit).all()
            or (self.limit <= 0).any()
            or not np.isfinite([tolerance, velocity_tolerance]).all()
            or tolerance <= 0
            or velocity_tolerance <= 0
        ):
            raise ValueError("Invalid waypoints, limits, or tolerances")
        self.tolerance, self.velocity_tolerance = tolerance, velocity_tolerance
        self.reset()

    def reset(self):
        self.index = 0

    @property
    def done(self):
        return self.index == len(self.waypoints)

    def predict(self, observation: Observation) -> ActionChunk:
        q, qd = np.asarray(observation.joint_position), np.asarray(observation.joint_velocity)
        if q.shape != self.limit.shape or qd.shape != q.shape or not np.isfinite([q, qd]).all():
            raise ValueError("Invalid joint observation")
        if (
            not self.done
            and (np.abs(self.waypoints[self.index] - q) <= self.tolerance).all()
            and (np.abs(qd) <= self.velocity_tolerance).all()
        ):
            self.index += 1
        delta = (
            np.zeros_like(q)
            if self.done
            else np.clip(self.waypoints[self.index] - q, -self.limit, self.limit)
        )
        return ActionChunk((tuple(delta),))
