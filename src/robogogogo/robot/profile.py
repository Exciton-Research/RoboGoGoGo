"""One dual-arm deployment identity; unset hardware is never a simulated identity."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CAMERA_ROLES = ("head_left", "head_right", "wrist_left", "wrist_right")


@dataclass(frozen=True)
class RobotProfile:
    name: str = "Dual-arm robot"
    model: str | None = None
    backend: str | None = None
    left_joints: tuple[str, ...] = ()
    right_joints: tuple[str, ...] = ()
    left_gripper: str | None = None
    right_gripper: str | None = None
    # Ordered physical stream names for the four fixed camera roles above.
    cameras: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported robot profile version")
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 128:
            raise ValueError("A robot name is required")
        for value in (self.model, self.backend, self.left_gripper, self.right_gripper):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 128
            ):
                raise ValueError("Hardware identifiers must be nonempty strings or null")
        for values in (self.left_joints, self.right_joints, self.cameras):
            if (
                not isinstance(values, tuple)
                or len(values) > 48
                or any(not isinstance(n, str) or not n or len(n) > 128 for n in values)
            ):
                raise ValueError("Profile mappings must be bounded tuples of names")
        joints = self.left_joints + self.right_joints
        if len(set(joints)) != len(joints):
            raise ValueError("Each arm must own distinct joints")
        if self.cameras and (len(self.cameras) != 4 or len(set(self.cameras)) != 4):
            raise ValueError("Map head stereo and both wrist cameras to four distinct streams")
        if self.left_gripper is not None and self.left_gripper == self.right_gripper:
            raise ValueError("Each arm requires its own gripper")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 65536:
            raise ValueError("Robot profile exceeds size limit")
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Robot profile must contain exactly the versioned profile fields")
        for key in ("left_joints", "right_joints", "cameras"):
            if not isinstance(value[key], list):
                raise ValueError(f"{key} must be a list")
            value[key] = tuple(value[key])
        return cls(**value)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def missing(self):
        return [
            name
            for name in (
                "model",
                "backend",
                "left_joints",
                "right_joints",
                "left_gripper",
                "right_gripper",
                "cameras",
            )
            if not getattr(self, name)
        ]

    def snapshot(self):
        return dict(
            **asdict(self),
            digest=self.digest,
            missing=self.missing(),
            camera_roles=list(CAMERA_ROLES),
        )
