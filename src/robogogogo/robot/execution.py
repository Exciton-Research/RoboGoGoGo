"""Coordinated execution contract, separate from read-only telemetry.

Adapters must implement bounded calls, atomic group acceptance, confirmed stopping,
and a controller-side lease timeout. This contract alone cannot certify hardware.
Joint positions are radians; gripper opening widths are meters and force caps newtons.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GripperGoal:
    width: float
    force_limit: float


@dataclass(frozen=True)
class CoordinatedGoal:
    left: tuple[float, ...]
    right: tuple[float, ...]
    left_gripper: GripperGoal
    right_gripper: GripperGoal
    duration_s: float


@dataclass(frozen=True)
class ExecutionFeedback:
    sequence: int
    timestamp: float
    left: tuple[float, ...]
    right: tuple[float, ...]
    left_velocity: tuple[float, ...]
    right_velocity: tuple[float, ...]
    grippers: tuple[float, float]
    command_id: str | None
    moving: bool
    fault: str | None = None


@dataclass(frozen=True)
class ExecutionLimits:
    joints_per_arm: int = 6
    joint_lower: float = -1.5
    joint_upper: float = 1.5
    joint_speed: float = 1.0
    joint_acceleration: float = 2.0
    gripper_width: float = 0.08
    gripper_speed: float = 0.08
    gripper_acceleration: float = 0.2
    gripper_force: float = 20.0


class ExecutionBackend(Protocol):
    """Future vendor adapters must honor these semantics, not just method names.

    No per-arm partial acceptance is permitted. Rejection after partial physical
    delivery must stop the whole group and report a fault. stop() must acknowledge
    actual controller stop; sending a stop message is not an acknowledgement.
    """

    is_test: bool
    limits: ExecutionLimits

    def read(self) -> ExecutionFeedback: ...
    def enable(self, lease_until: float) -> None: ...
    def renew(self, lease_until: float) -> None: ...
    def execute(self, command_id: str, goal: CoordinatedGoal) -> None: ...
    def stop(self) -> ExecutionFeedback: ...
    def close(self) -> None: ...
