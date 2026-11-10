"""Deterministic command test double. No motors, contacts, or force sensing.

Both arms and grippers share a quintic trajectory clock and atomic acceptance.
The backend's lease clamps motion even when the supervisor is not being ticked.
Fault switches are Python test hooks, not remotely exposed hardware controls.
"""

import threading
import time

from .execution import ExecutionFeedback, ExecutionLimits


class TestExecutionBackend:
    __test__ = False
    is_test = True

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.limits = ExecutionLimits()
        self.lock = threading.RLock()
        self.position = [0.0] * 12 + [0.06, 0.06]
        self.velocity = [0.0] * 14
        self.sequence = 0
        self.command_id = None
        self.trajectory = None
        self.lease_until = None
        self.fault = None
        self.closed = False
        self.freeze_feedback = self.reject_next = self.fail_stop = self.stall_motion = False
        self.cached = None
        self.accepted = []
        self.force_caps = (0.0, 0.0)

    def _advance(self):
        now = self.clock()
        effective = min(now, self.lease_until) if self.lease_until is not None else now
        if self.trajectory and not self.stall_motion:
            start, duration, origin, target = self.trajectory
            u = max(0.0, min(1.0, (effective - start) / duration))
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            speed = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
            self.position = [a + (b - a) * blend for a, b in zip(origin, target)]
            self.velocity = [(b - a) * speed for a, b in zip(origin, target)]
            if u >= 1:
                self.trajectory = None
        if self.lease_until is not None and now >= self.lease_until:
            self.trajectory = None
            self.velocity = [0.0] * 14
            self.lease_until = None
            self.fault = "Backend lease expired"

    def read(self):
        with self.lock:
            self._advance()
            if self.freeze_feedback and self.cached is not None:
                return self.cached
            self.sequence += 1
            self.cached = ExecutionFeedback(
                self.sequence,
                self.clock(),
                tuple(self.position[:6]),
                tuple(self.position[6:12]),
                tuple(self.velocity[:6]),
                tuple(self.velocity[6:12]),
                tuple(self.position[12:]),
                tuple(self.velocity[12:]),
                self.command_id,
                self.trajectory is not None,
                self.fault,
            )
            return self.cached

    def enable(self, lease_until):
        with self.lock:
            self._advance()
            if self.closed or self.trajectory or self.fault or lease_until <= self.clock():
                raise ValueError("Test backend is not ready to enable")
            self.lease_until = lease_until

    def renew(self, lease_until):
        with self.lock:
            self._advance()
            if self.closed or self.fault or self.lease_until is None or lease_until <= self.clock():
                raise ValueError("Cannot renew inactive backend lease")
            self.lease_until = lease_until

    def execute(self, command_id, goal):
        with self.lock:
            self._advance()
            if self.closed or self.fault or self.lease_until is None or self.trajectory:
                raise ValueError("Test backend cannot accept this command")
            if self.reject_next:
                self.reject_next = False
                raise ValueError("Injected atomic group rejection")
            target = list(goal.left + goal.right) + [
                goal.left_gripper.width,
                goal.right_gripper.width,
            ]
            self.trajectory = (self.clock(), goal.duration_s, list(self.position), target)
            self.command_id = command_id
            self.force_caps = (goal.left_gripper.force_limit, goal.right_gripper.force_limit)
            self.accepted.append(command_id)
            del self.accepted[:-128]

    def stop(self):
        with self.lock:
            self._advance()
            if self.fail_stop:
                raise RuntimeError("Injected unconfirmed stop")
            self.trajectory = None
            self.velocity = [0.0] * 14
            self.lease_until = None
            self.fault = None
            return self.read()

    def close(self):
        with self.lock:
            self.stop()
            self.closed = True
