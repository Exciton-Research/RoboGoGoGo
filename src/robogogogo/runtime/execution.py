"""Serialized command ownership, validation and watchdog supervision.

This is a soft-real-time host service. Hardware must implement bounded adapter
calls and an independent controller watchdog. Only the test backend is enabled
in Studio until a robot-specific execution adapter is commissioned.
"""

import math
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict
from uuid import uuid4

from robogogogo.robot.execution import CoordinatedGoal


class CommandExecutor:
    lease_s = 1.0
    feedback_timeout_s = 0.25
    observation_timeout_s = 0.25
    completion_grace_s = 0.3

    def __init__(self, backend, *, clock=time.monotonic):
        self.backend, self.clock = backend, clock
        self.lock = threading.RLock()
        self.mode = "disarmed"
        self.owner = self.token = None
        self.epoch = 0
        self.deadline = 0.0
        self.last_sequence = 0
        self.active = None
        self.feedback = None
        self.observations = deque(maxlen=64)
        self.events = deque(maxlen=128)
        self.stop_confirmed = False
        self.reason = "Explicit arming required"
        self.shutdown = threading.Event()
        self.thread = None
        self._halt("Startup stop", fault=False)

    def _event(self, kind, **details):
        self.events.append(dict(time=self.clock(), kind=kind, epoch=self.epoch, **details))

    def _validate_feedback(self, value):
        n = self.backend.limits.joints_per_arm
        numbers = (
            value.left + value.right + value.left_velocity + value.right_velocity + value.grippers
        )
        if (
            any(
                len(v) != n
                for v in (value.left, value.right, value.left_velocity, value.right_velocity)
            )
            or len(value.grippers) != 2
            or not all(math.isfinite(v) for v in numbers)
            or type(value.sequence) is not int
            or not math.isfinite(value.timestamp)
            or not 0 <= self.clock() - value.timestamp <= self.feedback_timeout_s
        ):
            raise ValueError("Invalid or stale execution feedback")
        limits = self.backend.limits
        if any(not limits.joint_lower <= q <= limits.joint_upper for q in value.left + value.right):
            raise ValueError("Measured arm position exceeds execution limits")
        if any(not 0 <= width <= limits.gripper_width for width in value.grippers):
            raise ValueError("Measured gripper width exceeds execution limits")
        if value.fault:
            raise ValueError(value.fault)
        return value

    def _read(self):
        value = self._validate_feedback(self.backend.read())
        if self.feedback and value.sequence < self.feedback.sequence:
            raise ValueError("Execution feedback sequence rolled back")
        if self.feedback and value.sequence == self.feedback.sequence:
            if value != self.feedback:
                raise ValueError("Conflicting execution feedback sequence")
        else:
            self.feedback = value
            self.observations.append((value.sequence, value.timestamp))
        return value

    def _halt(self, reason, *, fault):
        # Revoke authority before any potentially failing driver operation.
        self.epoch += 1
        self.token = self.owner = None
        self.deadline = 0.0
        interrupted = self.active[0] if self.active else None
        self.active = None
        self.stop_confirmed = False
        self.mode = "fault" if fault else "stopped"
        self.reason = reason
        try:
            old_sequence = self.feedback.sequence if self.feedback else -1
            value = self._validate_feedback(self.backend.stop())
            if (
                value.sequence <= old_sequence
                or value.moving
                or any(abs(v) > 1e-6 for v in value.left_velocity + value.right_velocity)
            ):
                raise ValueError("Backend did not acknowledge a fresh stationary stop")
            self.feedback = value
            self.observations.clear()
            self.observations.append((value.sequence, value.timestamp))
            self.stop_confirmed = True
        except Exception as error:
            self.mode = "fault"
            self.reason = f"{reason}; stop unconfirmed: {error}"
        self._event(
            "stop", reason=self.reason, confirmed=self.stop_confirmed, interrupted=interrupted
        )

    def _check(self):
        if self.mode in ("fault", "closed"):
            return
        now = self.clock()
        if self.token and now >= self.deadline:
            self._halt("Owner heartbeat watchdog expired", fault=True)
            return
        try:
            feedback = self._read()
            if self.active:
                identifier, goal, finish = self.active
                if feedback.command_id != identifier:
                    raise ValueError("Execution acknowledgement does not match active command")
                target = (
                    goal.left + goal.right + (goal.left_gripper.width, goal.right_gripper.width)
                )
                actual = feedback.left + feedback.right + feedback.grippers
                if not feedback.moving:
                    if any(abs(a - b) > 1e-4 for a, b in zip(target, actual)):
                        raise ValueError("Command stopped before reaching coordinated targets")
                    if any(abs(v) > 1e-4 for v in feedback.left_velocity + feedback.right_velocity):
                        raise ValueError("Command completion has nonzero arm velocity")
                    self.active = None
                    self.mode = "armed"
                    self._event("completed", command_id=identifier)
                elif now > finish + self.completion_grace_s:
                    raise ValueError("Command completion watchdog expired")
            elif feedback.moving:
                raise ValueError("Unexpected motion without an active command")
        except Exception as error:
            self._halt(f"Execution fault: {error}", fault=True)

    def start(self):
        with self.lock:
            if self.thread is not None or self.mode == "closed":
                raise ValueError("Execution service cannot be restarted")
            self.thread = threading.Thread(target=self._run, daemon=True, name="command-watchdog")
            self.thread.start()

    def _run(self):
        while not self.shutdown.wait(0.02):
            with self.lock:
                self._check()

    def tick(self):
        """Deterministic watchdog step for tests and embedded hosts."""
        with self.lock:
            self._check()

    def _authorize(self, token):
        self._check()
        if (
            not isinstance(token, str)
            or self.token is None
            or not secrets.compare_digest(token, self.token)
        ):
            raise ValueError("Control lease is missing, expired, or revoked")

    def _grant(self, owner):
        if owner not in ("manual", "policy"):
            raise ValueError("Owner must be manual or policy")
        if self.mode not in ("stopped", "disarmed") or not self.stop_confirmed:
            raise ValueError("Confirm stop and reset faults before arming")
        try:
            self._read()
            self.backend.enable(self.clock() + self.lease_s)
        except Exception as error:
            self._halt(f"Enable failed: {error}", fault=True)
            raise ValueError(self.reason) from error
        self.epoch += 1
        self.token = secrets.token_urlsafe(32)
        self.owner = owner
        self.deadline = self.clock() + self.lease_s
        self.last_sequence = 0
        self.mode, self.reason = "armed", f"{owner} owns the control lease"
        self.stop_confirmed = False
        self._event("armed", owner=owner)
        return dict(
            token=self.token, epoch=self.epoch, heartbeat_interval_s=0.25, snapshot=self.snapshot()
        )

    def arm(self, owner):
        with self.lock:
            self._check()
            return self._grant(owner)

    def heartbeat(self, token):
        with self.lock:
            self._authorize(token)
            deadline = self.clock() + self.lease_s
            try:
                self.backend.renew(deadline)
            except Exception as error:
                self._halt(f"Lease renewal failed: {error}", fault=True)
                raise ValueError(self.reason) from error
            self.deadline = deadline
            return self.snapshot()

    def takeover(self):
        with self.lock:
            self._check()
            if self.mode in ("closed", "fault"):
                raise ValueError("Reset execution faults before manual takeover")
            self._halt("Manual takeover requested", fault=False)
            return self._grant("manual")

    def stop(self):
        with self.lock:
            if self.mode != "closed":
                self._halt("Operator stop", fault=self.mode == "fault")
            return self.snapshot()

    def reset(self):
        with self.lock:
            if self.mode not in ("fault", "stopped", "disarmed"):
                raise ValueError("Stop execution before resetting")
            self._halt("Explicit fault reset", fault=False)
            if not self.stop_confirmed:
                raise ValueError(self.reason)
            return self.snapshot()

    def _validate_goal(self, goal):
        limits = self.backend.limits
        if not isinstance(goal, CoordinatedGoal) or any(
            len(v) != limits.joints_per_arm for v in (goal.left, goal.right)
        ):
            raise ValueError("A complete target for each arm is required")
        values = (
            goal.left
            + goal.right
            + (
                goal.duration_s,
                goal.left_gripper.width,
                goal.right_gripper.width,
                goal.left_gripper.force_limit,
                goal.right_gripper.force_limit,
            )
        )
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("Command values must be finite numbers")
        if not 0.1 <= goal.duration_s <= 10:
            raise ValueError("Duration must be between 0.1 and 10 seconds")
        for q in goal.left + goal.right:
            if not limits.joint_lower <= q <= limits.joint_upper:
                raise ValueError("Joint target exceeds position limits")
        for gripper in (goal.left_gripper, goal.right_gripper):
            if (
                not 0 <= gripper.width <= limits.gripper_width
                or not 0 < gripper.force_limit <= limits.gripper_force
            ):
                raise ValueError("Gripper width or force limit exceeds bounds")
        # Quintic interpolation has peak normalized speed 1.875 and acceleration 5.774.
        # Conservative rounded bounds apply identically to each arm and each gripper.
        origin = self.feedback.left + self.feedback.right + self.feedback.grippers
        target = goal.left + goal.right + (goal.left_gripper.width, goal.right_gripper.width)
        for index, (a, b) in enumerate(zip(origin, target)):
            gripper = index >= 2 * limits.joints_per_arm
            speed = limits.gripper_speed if gripper else limits.joint_speed
            acceleration = limits.gripper_acceleration if gripper else limits.joint_acceleration
            distance = abs(a - b)
            if (
                1.875 * distance / goal.duration_s > speed
                or 5.774 * distance / goal.duration_s**2 > acceleration
            ):
                raise ValueError("Command duration violates speed or acceleration limits")

    def submit(self, token, sequence, observation_sequence, goal):
        with self.lock:
            self._authorize(token)
            if self.active is not None:
                raise ValueError(
                    "A coordinated command is already executing; stop before replacing it"
                )
            if type(sequence) is not int or sequence != self.last_sequence + 1:
                raise ValueError("Command sequence must advance exactly once")
            observed = dict(self.observations).get(observation_sequence)
            if observed is None or not 0 <= self.clock() - observed <= self.observation_timeout_s:
                raise ValueError("Command references missing or stale observation")
            self._validate_goal(goal)
            identifier = str(uuid4())
            try:
                self.backend.execute(identifier, goal)
                self.active = (identifier, goal, self.clock() + goal.duration_s)
                self.last_sequence = sequence
                self.mode, self.reason = "executing", "Coordinated arm and gripper command"
                self._event(
                    "accepted",
                    command_id=identifier,
                    owner=self.owner,
                    sequence=sequence,
                    goal=asdict(goal),
                )
                self._check()
                if self.mode == "fault":
                    raise ValueError(self.reason)
            except Exception as error:
                if self.mode != "fault":
                    self._halt(f"Command acceptance failed: {error}", fault=True)
                raise ValueError(self.reason) from error
            return dict(command_id=identifier, snapshot=self.snapshot())

    def snapshot(self):
        with self.lock:
            self._check()
            return dict(
                backend="test" if self.backend.is_test else "physical",
                physical_execution_enabled=not self.backend.is_test,
                mode=self.mode,
                owner=self.owner,
                epoch=self.epoch,
                reason=self.reason,
                stop_confirmed=self.stop_confirmed,
                lease_remaining_s=max(0, self.deadline - self.clock()) if self.token else 0,
                next_sequence=self.last_sequence + 1,
                active_command=self.active[0] if self.active else None,
                feedback=asdict(self.feedback) if self.feedback else None,
                limits=asdict(self.backend.limits),
                events=list(self.events),
            )

    def close(self):
        self.shutdown.set()
        if self.thread:
            self.thread.join(timeout=2)
        with self.lock:
            self._halt("Execution service shutdown", fault=False)
            try:
                self.backend.close()
            except Exception as error:
                self.mode, self.reason, self.stop_confirmed = "fault", str(error), False
                raise RuntimeError("Execution backend shutdown is unconfirmed") from error
            self.mode = "closed"
