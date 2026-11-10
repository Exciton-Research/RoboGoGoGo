"""Validated HTTP and persisted-checkpoint schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Command(Payload):
    op: Literal[
        "run",
        "pause",
        "step",
        "reset",
        "home",
        "joints",
        "target",
        "scene",
        "speed",
        "posture",
        "drive",
        "brake",
        "cancel_plan",
    ]
    start: bool = False
    duration: float = Field(default=3.0, gt=0, le=5)
    friction: float | None = Field(default=None, ge=0, le=1.5)
    posture: Literal["standing", "low", "reach"] | None = None
    arm: str | None = None
    values: list[float] | None = None
    scene: Literal["arm", "dual_arm", "dual_arm_head", "mobile_dual_arm", "drop"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=2.0)


class ArmSpec(Payload):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,79}$")
    tool: str
    root: str | None = None


class ImportRobot(Payload):
    arms: list[ArmSpec] | None = Field(default=None, min_length=1, max_length=16)
    xml: str = Field(min_length=1, max_length=262144)


class SaveProject(Payload):
    name: str = Field(min_length=1, max_length=80)
    id: str | None = None


class CameraSpec(Payload):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    width: int = Field(default=320, ge=32, le=640)
    height: int = Field(default=240, ge=32, le=480)
    mount: str | None = None
    position: tuple[float, float, float]
    look_at: tuple[float, float, float] | None = None
    rpy: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0, 0, 1)
    vertical_fov_degrees: float = Field(default=45, gt=1, lt=170)
    near: float = Field(default=0.01, gt=0)
    far: float = Field(default=50, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self):
        import numpy as np

        if self.rpy is not None:
            from robogogogo.robot.cameras import mounting_vectors

            forward, up = mounting_vectors(self.rpy)
            self.look_at = tuple(np.asarray(self.position) + forward)
            self.up = tuple(up)
        if self.look_at is None:
            raise ValueError("Supply mount-relative rpy or a legacy look_at direction")
        direction = np.asarray(self.look_at) - self.position
        if self.far <= self.near or np.linalg.norm(np.cross(direction, self.up)) < 1e-8:
            raise ValueError(
                "Camera requires valid clipping planes and a nondegenerate view direction/up"
            )
        return self


class NewEpisode(Payload):
    instruction: str = Field(
        default="Move the robot tool to the highlighted target.", min_length=1, max_length=1000
    )
    variant: Literal["a", "b", "c"] = "a"
    camera_preset: Literal["head_wrists", "head"] | None = None
    cameras: list[CameraSpec] | None = Field(default=None, min_length=1, max_length=8)


class PlanTask(Payload):
    active_perception: bool = False
    camera_recovery: bool = False
    structural_dynamics: bool = False
    recovery_behaviors: bool = False
    object_material: Literal["rigid_reference", "soft_foam", "soft_elastomer", "fragile_shell"] = (
        "rigid_reference"
    )
    adaptive_grasp: bool = True
    initial_grip_force_n: float = Field(default=24.0, gt=0.0, le=80.0)
    object_damage_force_n: float = Field(default=55.0, gt=0.0, le=200.0)
    grip_force_n: float = Field(default=40.0, ge=0.0, le=80.0)
    friction_static: float = Field(default=0.8, ge=0.0, le=2.0)
    friction_dynamic: float = Field(default=0.6, ge=0.0, le=2.0)
    kind: Literal["inspection", "handover", "material_grasp"] = "inspection"
    station: tuple[float, float, float] = (0.55, 0, 0.75)
    seed: int = Field(default=7, ge=0, le=1000000)

    @model_validator(mode="after")
    def validate_friction(self):
        if self.active_perception and not self.camera_recovery:
            raise ValueError("Active perception requires camera-based recovery")
        if self.camera_recovery and not self.recovery_behaviors:
            raise ValueError("Camera recovery requires coordinated recovery")
        if self.camera_recovery and self.kind == "inspection":
            raise ValueError("Camera recovery requires a manipulation task")
        if self.recovery_behaviors and not self.adaptive_grasp:
            raise ValueError("Recovery requires adaptive force feedback")
        if self.structural_dynamics and self.object_material == "rigid_reference":
            raise ValueError("Structural dynamics require a deformable material")
        if self.adaptive_grasp and self.initial_grip_force_n > 0.75 * self.object_damage_force_n:
            raise ValueError("Initial grip exceeds reserved object force envelope")
        if self.friction_dynamic > self.friction_static:
            raise ValueError("Sliding friction must not exceed static friction")
        return self


class TaskStep(Payload):
    observation: int = Field(ge=0)


class StereoRequest(Payload):
    left: str = "head_left"
    right: str = "head_right"
    observation: int = Field(ge=0)


class CameraObservation(Payload):
    ground_truth_depth: dict[str, str] | None = Field(default=None, min_length=1, max_length=8)
    image: str | None = Field(default=None, max_length=700000)
    images: dict[str, str] | None = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def one_capture(self):
        if (self.image is None) == (self.images is None):
            raise ValueError("Supply exactly one of image or images")
        if self.images and any(len(value) > 700000 for value in self.images.values()):
            raise ValueError("Camera image is oversized")
        return self

    sequence: int


class PolicyAction(Payload):
    delta: list[float]
    observation: int = Field(ge=0)
    control_ticks: int = Field(default=10, ge=1, le=20)
    source: Literal["manual", "scripted_reference", "policy", "rule_based", "task_planner"] = (
        "manual"
    )


class Project(Payload):
    tire_friction: float = Field(default=0.7, ge=0, le=1.5)
    mobile_com_acceleration: tuple[float, float, float] = (0, 0, 0)
    format: Literal["robogogogo-project-v1"] = "robogogogo-project-v1"
    id: str
    name: str = Field(min_length=1, max_length=80)
    updated: str
    scene: Literal["arm", "dual_arm", "dual_arm_head", "mobile_dual_arm", "drop", "urdf"]
    xml: str | None = Field(default=None, max_length=262144)
    dt: float = Field(gt=0, le=0.005)
    substeps: int = Field(ge=1, le=20)
    q: list[float]
    qd: list[float]
    goal: list[float]
    target: list[float] = Field(min_length=3, max_length=3)
    arms: list[ArmSpec] | None = None
    targets: dict[str, tuple[float, float, float]] | None = None
    elapsed: float = Field(ge=0)
    cameras: list[CameraSpec] | None = Field(default=None, min_length=1, max_length=8)
    instruction: str = Field(
        default="Move the robot tool to the highlighted target.", max_length=1000
    )
