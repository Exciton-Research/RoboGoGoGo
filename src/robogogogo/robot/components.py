"""Engineering component selections from manufacturer data, checked 2026-11-10.

Selections are not a purchasing release. Aggregate cylinder geometry excludes
connector protrusions and vendor-specific fasteners; brackets are custom designs.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Actuator:
    model: str
    diameter_m: float
    length_m: float
    mass_kg: float
    ratio: int
    rated_nm: float
    peak_nm: float
    no_load_rpm: float
    voltage: int
    source: str

    def to_dict(self):
        return asdict(self)


AK80 = Actuator(
    "CubeMars AK80-64 KV80",
    0.098,
    0.0619,
    0.850,
    64,
    48,
    120,
    75,
    48,
    "https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html",
)
AK60 = Actuator(
    "CubeMars AK60-39 V3.0 KV80",
    0.079,
    0.067,
    0.750,
    39,
    24,
    72,
    98,
    48,
    "https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html",
)
AK45 = Actuator(
    "CubeMars AK45-36 V3.0 KV80",
    0.055,
    0.0565,
    0.349,
    36,
    8,
    24,
    52,
    24,
    "https://www.cubemars.com/product/ak45-36-v3-0-kv80-robotic-actuator.html",
)
ARM_COMPONENTS = {
    "waist": AK60,
    "upper_roll": AK80,
    "shoulder": AK80,
    "elbow": AK60,
    "wrist_roll": AK45,
    "wrist_pitch": AK45,
    "tool_roll": AK45,
}


def mounting_offset(suffix):
    # Catalog assemblies are mounted outside the intersecting axes; a custom
    # output coupling/yoke bridges to the joint. Connectors need separate clearance.
    return {"waist": 0.13, "upper_roll": 0.083, "shoulder": 0.12}.get(suffix, 0.025)


def mounting_sign(side, suffix):
    if suffix in ("wrist_roll", "upper_roll"):
        return -1
    return -1 if side == "right" and suffix in ("shoulder", "elbow", "wrist_pitch") else 1


def mounting_mass(suffix):
    """Mass of modeled 8 mm aluminum saddle rods and 12 g connector estimate."""
    from math import pi

    spec = ARM_COMPONENTS[suffix]
    length = 2 * (spec.length_m + 0.012) + 8 * (spec.diameter_m / 2 + 0.008)
    return pi * 0.004**2 * (length + 0.012) * 2700 + 0.012 + 2 * pi * 0.003**2 * 0.03 * 2700 + 0.005
