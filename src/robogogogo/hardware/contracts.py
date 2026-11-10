"""Read-only robot telemetry. No command/actuation methods are defined here."""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class JointSample:
    timestamp: float
    names: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...] = ()
    effort: tuple[float, ...] = ()
    frame_id: str = ""
    clock_domain: str = ""


@dataclass(frozen=True)
class CameraSample:
    name: str
    timestamp: float
    data: bytes
    media_type: str
    frame_id: str = ""
    clock_domain: str = ""


@dataclass(frozen=True)
class CameraInfoSample:
    name: str
    frame_id: str
    width: int
    height: int
    intrinsics: tuple
    distortion: tuple
    distortion_model: str


@dataclass(frozen=True)
class TelemetrySink:
    joints: Callable[[JointSample], None]
    camera: Callable[[CameraSample], None]
    error: Callable[[str], None]
    calibration: Callable[[CameraInfoSample], None] | None = None


class ReadOnlyBackend(Protocol):
    def start(self, sink: TelemetrySink) -> None:
        """Start subscriptions or a bounded background reader; return promptly."""

    def close(self) -> None:
        """Release owned subscriptions/connections; never send a motion command."""
