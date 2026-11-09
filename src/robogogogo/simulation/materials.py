"""Body-local viscoelastic indentation foundation with irreversible crush/damage.

A reduced-order surface model, not volumetric elasticity or fragment simulation.
SI parameters describe illustrative proxies, not measured commercial materials.
"""

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

AXIAL_CELLS = 32
ANGULAR_CELLS = 24
CELLS = AXIAL_CELLS * ANGULAR_CELLS


@dataclass(frozen=True)
class Material:
    name: str
    modulus_pa: float
    thickness_m: float
    relaxation_s: float
    yield_strain: float
    failure_strain: float
    strength_pa: float
    plastic_rate_s: float
    mass_kg: float
    initial_force_n: float
    force_threshold_n: float
    color: str

    def __post_init__(self):
        numbers = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError("Material parameters must be positive and finite")
        if not 0 < self.yield_strain < self.failure_strain < 1:
            raise ValueError("Require 0 < yield strain < failure strain < 1")
        if self.thickness_m > 0.025 or self.initial_force_n > 0.75 * self.force_threshold_n:
            raise ValueError("Invalid foundation depth or initial force envelope")


MATERIALS = {
    "soft_foam": Material(
        "soft_foam", 180_000, 0.015, 0.04, 0.28, 0.65, 110_000, 6, 0.04, 4, 10, "#77b8ad"
    ),
    "soft_elastomer": Material(
        "soft_elastomer", 1_200_000, 0.012, 0.015, 0.65, 0.85, 1_100_000, 2, 0.16, 16, 40, "#9d98d1"
    ),
    "fragile_shell": Material(
        "fragile_shell", 15_000_000, 0.004, 0.002, 0.006, 0.025, 180_000, 15, 0.06, 6, 12, "#e6cf9d"
    ),
}


def material_catalog():
    return {name: asdict(material) for name, material in MATERIALS.items()}


class MaterialState(NamedTuple):
    indentation: jax.Array
    permanent: jax.Array
    damage: jax.Array
    pressure: jax.Array
    peak_pressure: jax.Array
    peak_strain: jax.Array
    risk: jax.Array


def initial_material_state(dtype=jnp.float64):
    return MaterialState(*(jnp.zeros(CELLS, dtype=dtype) for _ in range(7)))


def surface_indices(local_points, half_length, radius):
    """Nearest body-local cell; cap ends share the end axial bands."""
    axial = jnp.clip(
        jnp.floor(
            (local_points[:, 1] + half_length + radius) / (2 * (half_length + radius)) * AXIAL_CELLS
        ),
        0,
        AXIAL_CELLS - 1,
    ).astype(jnp.int32)
    angle = jnp.mod(jnp.arctan2(local_points[:, 2], local_points[:, 0]), 2 * jnp.pi)
    angular = jnp.floor(angle * ANGULAR_CELLS / (2 * jnp.pi)).astype(jnp.int32)
    return axial * ANGULAR_CELLS + jnp.minimum(angular, ANGULAR_CELLS - 1)


def surface_stencil(local_points, half_length, radius):
    """Periodic bilinear interpolation over cell centers; avoids moving-contact jumps."""
    y = jnp.clip(
        (local_points[:, 1] + half_length + radius) / (2 * (half_length + radius)) * AXIAL_CELLS
        - 0.5,
        0,
        AXIAL_CELLS - 1,
    )
    angle = jnp.mod(jnp.arctan2(local_points[:, 2], local_points[:, 0]), 2 * jnp.pi)
    a = angle * ANGULAR_CELLS / (2 * jnp.pi) - 0.5
    iy, ia = jnp.floor(y).astype(jnp.int32), jnp.floor(a).astype(jnp.int32)
    wy, wa = y - iy, a - ia
    rows = jnp.stack(
        (iy, iy, jnp.minimum(iy + 1, AXIAL_CELLS - 1), jnp.minimum(iy + 1, AXIAL_CELLS - 1)), axis=1
    )
    cols = jnp.stack((ia, ia + 1, ia, ia + 1), axis=1) % ANGULAR_CELLS
    weights = jnp.stack(((1 - wy) * (1 - wa), (1 - wy) * wa, wy * (1 - wa), wy * wa), axis=1)
    return rows * ANGULAR_CELLS + cols, weights


def material_response(
    state, material, indices, overlap, normal_velocity, area, pad_k, pad_c, dt, weights=None
):
    """Implicit Kelvin–Voigt foundation in series with each contact pad spring.

    Multiple samples of a cell share its history; max-envelope scatter prevents
    double-counting indentation. This is a sampled foundation, not a surface PDE.
    No friction oracle, pose attachment, or clipping of measured normal loads.
    """

    def gather(field):
        return field[indices] if weights is None else jnp.sum(field[indices] * weights, axis=1)

    def scatter(field, values):
        if weights is None:
            return field.at[indices].max(values)
        values = jnp.where(weights > 0, values[:, None], 0.0)
        return field.at[indices.reshape(-1)].max(values.reshape(-1))

    p = gather(state.permanent)
    old = gather(state.indentation)
    stiffness = material.modulus_pa * area / material.thickness_m
    stiffness *= 1 - 0.9 * gather(state.damage)
    viscosity = stiffness * material.relaxation_s
    free = (stiffness * p + viscosity / dt * old) / (stiffness + viscosity / dt)
    active = overlap > free
    combined = viscosity + pad_c
    x = (pad_k * overlap - pad_c * normal_velocity + stiffness * p + combined / dt * old) / (
        pad_k + stiffness + combined / dt
    )
    x = jnp.where(active, jnp.maximum(p, x), free)
    fn = jnp.where(
        active,
        jnp.maximum(pad_k * (overlap - x) - pad_c * (normal_velocity + (x - old) / dt), 0),
        0,
    )
    # Perzyna-style irreversible compression flow; permanent dents stay with the object.
    plastic = jnp.minimum(
        x,
        p
        + dt
        * material.plastic_rate_s
        * jnp.maximum(x - p - material.yield_strain * material.thickness_m, 0),
    )
    strain = x / material.thickness_m
    pressure = fn / area
    risk = jnp.maximum(strain / material.failure_strain, pressure / material.strength_pa)
    damage = jnp.clip((risk - 0.7) / 0.3, 0, 1)
    # Warn before irreversible yielding as well as before pressure/strain damage.
    protection_risk = jnp.maximum(
        risk, 0.7 * jnp.maximum(x - p, 0) / (material.yield_strain * material.thickness_m)
    )
    # Unloaded cells recover toward their permanent set, never toward an undeformed reset.
    relaxed = state.permanent + (state.indentation - state.permanent) / (
        1 + dt / material.relaxation_s
    )
    indentation = scatter(relaxed, x)
    permanent = scatter(state.permanent, plastic)
    current_pressure = scatter(jnp.zeros_like(state.pressure), pressure)
    next_state = MaterialState(
        indentation,
        permanent,
        scatter(state.damage, damage),
        current_pressure,
        jnp.maximum(state.peak_pressure, current_pressure),
        scatter(state.peak_strain, strain),
        scatter(jnp.zeros_like(state.risk), protection_risk),
    )
    return fn, next_state


def material_summary(material, state):
    if material is None:
        return dict(
            name="rigid_reference",
            model="rigid_with_compliant_pads",
            damaged=False,
            fractured=False,
        )
    permanent = np.asarray(state.permanent)
    damage = np.asarray(state.damage)
    return dict(
        name=material.name,
        model="viscoelastic_surface_foundation_v1",
        parameters=asdict(material),
        indentation_m=np.asarray(state.indentation).reshape(AXIAL_CELLS, ANGULAR_CELLS).tolist(),
        permanent_m=permanent.reshape(AXIAL_CELLS, ANGULAR_CELLS).tolist(),
        damage=damage.reshape(AXIAL_CELLS, ANGULAR_CELLS).tolist(),
        maximum_indentation_m=float(np.max(state.indentation)),
        maximum_permanent_m=float(permanent.max()),
        damage_fraction=float(damage.max()),
        damaged=bool(damage.max() > 0 or permanent.max() > 1e-6),
        fractured=bool(damage.max() >= 1),
        peak_pressure_pa=float(np.max(state.peak_pressure)),
        peak_strain=float(np.max(state.peak_strain)),
        risk=float(np.max(state.risk)),
        sensor="simulated material state; requires an estimator on hardware",
    )
