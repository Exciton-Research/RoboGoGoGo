"""Study specification and paired, disjoint random streams. No simulator imports."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

REFERENCE = {
    "contact_friction_scale": 1.0,
    "camera_x_offset_m": 0.0,
    "light_scale": 1.0,
    "action_delay_steps": 0,
}
STRATEGIES = ("biased", "reference", "uniform", "random", "guided")


def seed_for(*parts):
    """Stable uint32 seed; independent of Python's randomized hash()."""
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


@dataclass(frozen=True)
class Study:
    name: str = "panda-lift-fidelity"
    seeds: tuple[int, ...] = (11, 23, 47)
    budget_rollouts: int = 48
    pilot_rollouts: int = 3
    selection_rollouts: int = 3
    evaluation_rollouts: int = 20
    horizon: int = 200
    control_hz: int = 20
    image_size: int = 64
    success_hold_steps: int = 10
    biased: dict = field(
        default_factory=lambda: {
            "contact_friction_scale": 0.4,
            "camera_x_offset_m": 0.04,
            "light_scale": 0.5,
            "action_delay_steps": 2,
        }
    )
    trainer: str = "robogogogo.research.policy:train_nearest"
    loader: str = "robogogogo.research.policy:load_nearest"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ValueError("seeds must be a nonempty list")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds must be uint32 integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        for key in (
            "budget_rollouts",
            "pilot_rollouts",
            "selection_rollouts",
            "evaluation_rollouts",
            "horizon",
            "control_hz",
            "image_size",
            "success_hold_steps",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.image_size < 8 or self.image_size % 8:
            raise ValueError("image_size must be a multiple of 8")
        if self.success_hold_steps > self.horizon:
            raise ValueError("success_hold_steps must not exceed horizon")
        if not isinstance(self.biased, dict) or set(self.biased) != set(REFERENCE):
            raise ValueError(f"biased must define exactly {tuple(REFERENCE)}")
        for key, value in self.biased.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 < self.biased["contact_friction_scale"] <= 5:
            raise ValueError("contact_friction_scale must be in (0, 5]")
        if not 0 < self.biased["light_scale"] <= 2:
            raise ValueError("light_scale must be in (0, 2]")
        if abs(self.biased["camera_x_offset_m"]) > 0.2:
            raise ValueError("camera_x_offset_m magnitude must be <= 0.2 m")
        delay = self.biased["action_delay_steps"]
        if type(delay) is not int or not 0 <= delay < self.horizon:
            raise ValueError("action_delay_steps must be an integer in [0, horizon)")
        if not self.axes:
            raise ValueError("At least one biased parameter must differ from reference")
        for key in ("trainer", "loader"):
            value = getattr(self, key)
            if not isinstance(value, str) or value.count(":") != 1 or not all(value.split(":")):
                raise ValueError(f"{key} must be an importable module:function")
        if self.budget_rollouts <= self.guided_overhead:
            raise ValueError(
                f"budget_rollouts must exceed guided pilot/selection cost {self.guided_overhead}"
            )

    @property
    def axes(self):
        return tuple(key for key in REFERENCE if self.biased[key] != REFERENCE[key])

    @property
    def guided_overhead(self):
        return (len(self.axes) + 1) * (self.pilot_rollouts + self.selection_rollouts)

    def profile(self, refined=()):
        unknown = set(refined) - set(self.axes)
        if unknown:
            raise ValueError(f"Unknown refinement axes: {sorted(unknown)}")
        return {key: REFERENCE[key] if key in refined else self.biased[key] for key in REFERENCE}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("Study specification must be an object")
        return cls(**payload)

    def plan(self):
        per_seed = {}
        for strategy in STRATEGIES:
            selection = (
                (len(self.axes) + 1) * self.selection_rollouts if strategy == "guided" else 0
            )
            per_seed[strategy] = {
                "training_rollouts": self.budget_rollouts - selection,
                "selection_rollouts": selection,
                "budgeted_control_steps": self.budget_rollouts * self.horizon,
                "held_out_rollouts": self.evaluation_rollouts,
            }
        return {
            "study": self.to_dict(),
            "active_axes": list(self.axes),
            "backend": "robosuite 1.5.1 / MuJoCo / Panda / Lift",
            "target": "nominal simulator, not a physical robot",
            "budget_unit": "fixed-horizon simulator rollout (collection + selection)",
            "strategies_per_seed": per_seed,
            "total_rollouts": len(self.seeds)
            * len(STRATEGIES)
            * (self.budget_rollouts + self.evaluation_rollouts),
            "excluded_from_budget": ["policy training compute", "common held-out evaluation"],
        }
