"""Composable software control boundary for learned, classical and hybrid policies.

The supervisor validates proposals before a caller sends them to simulation.
It is not a real-time or hardware-certified safety controller.
"""

from dataclasses import dataclass

import numpy as np

from robogogogo.contracts import ActionChunk, Observation, Policy


@dataclass(frozen=True)
class ControlDecision:
    action: ActionChunk
    controller: str
    limited: bool


class ControlSystem:
    """Select an explicit controller and enforce joint/action bounds on its proposal.

    Perception and task planning run upstream and populate Observation. Physics / PD
    actuation run downstream. A VLA adapter is a Policy just like a rule-based policy.
    Stop rejects proposals; it does not claim a zero delta can brake real hardware.
    """

    def __init__(
        self, controllers: dict[str, Policy], *, lower, upper, max_delta, collision_world=None
    ):
        self.controllers = dict(controllers)
        self.collision_world = collision_world
        self.lower, self.upper, self.limit = (
            np.asarray(x, dtype=float) for x in (lower, upper, max_delta)
        )
        if (
            self.lower.ndim != 1
            or not self.lower.size
            or self.upper.shape != self.lower.shape
            or self.limit.shape != self.lower.shape
            or np.isnan(self.lower).any()
            or np.isnan(self.upper).any()
            or np.isposinf(self.lower).any()
            or np.isneginf(self.upper).any()
            or not np.isfinite(self.limit).all()
            or (self.lower > self.upper).any()
            or (self.limit <= 0).any()
            or not self.controllers
        ):
            raise ValueError("Invalid control bounds or empty controller registry")
        self.stopped = False

    def stop(self):
        self.stopped = True

    def resume(self):
        self.stopped = False

    def step(self, observation: Observation, *, controller: str) -> ControlDecision:
        if self.stopped:
            raise ValueError("Control system is stopped")
        if controller not in self.controllers:
            raise ValueError(f"Unknown controller: {controller}")
        q = np.asarray(observation.joint_position, dtype=float)
        if (
            q.shape != self.lower.shape
            or not np.isfinite(q).all()
            or ((q < self.lower) | (q > self.upper)).any()
        ):
            raise ValueError("Invalid or out-of-bounds observed joint state")
        proposal = self.controllers[controller].predict(observation)
        actions = np.asarray(proposal.actions, dtype=float)
        if (
            proposal.representation != "joint_position_delta"
            or not isinstance(proposal.control_ticks, int)
            or not 1 <= proposal.control_ticks <= 20
            or actions.ndim != 2
            or not 1 <= len(actions) <= 256
            or actions.shape[1:] != q.shape
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Controller returned an invalid action chunk")
        bounded = []
        for delta in actions:
            next_q = np.clip(q + np.clip(delta, -self.limit, self.limit), self.lower, self.upper)
            if self.collision_world is not None and not self.collision_world.edge_free(q, next_q):
                raise ValueError("Controller proposal violates collision clearance")
            bounded.append(tuple(next_q - q))
            q = next_q
        limited = not np.allclose(actions, bounded, rtol=0, atol=1e-12)
        return ControlDecision(
            ActionChunk(tuple(bounded), proposal.control_ticks), controller, limited
        )
