"""Model-independent VLA observation, camera and action contracts."""

from dataclasses import dataclass, field
from typing import Protocol

CAMERA = dict(
    name="agent_view",
    width=320,
    height=240,
    channels=3,
    encoding="png",
    color_space="sRGB",
    position=[1.3, -1.6, 1.25],
    look_at=[0.35, 0.0, 0.4],
    up=[0.0, 0.0, 1.0],
    vertical_fov_degrees=45.0,
    near=0.01,
    far=50.0,
    renderer="Three.js WebGL rasterizer",
)


@dataclass(frozen=True)
class Observation:
    instruction: str
    image_png: bytes
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    time: float
    images: dict[str, bytes] = field(default_factory=dict)
    cameras: tuple[dict, ...] = ()
    camera_poses: dict[str, dict] = field(default_factory=dict)
    arms: tuple[dict, ...] = ()
    tool_positions: dict[str, list[float]] = field(default_factory=dict)
    targets: dict[str, list[float]] = field(default_factory=dict)
    ground_truth_depth: dict = field(default_factory=dict)
    manipulation: dict | None = None


@dataclass(frozen=True)
class ActionChunk:
    """Joint-position increments in radians/meters, one row per control action."""

    actions: tuple[tuple[float, ...], ...]
    control_ticks: int = 10
    representation: str = "joint_position_delta"


class Policy(Protocol):
    """Training/inference packages implement this interface without the Studio UI."""

    def predict(self, observation: Observation) -> ActionChunk: ...
