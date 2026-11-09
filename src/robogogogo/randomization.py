"""JAX-compatible physical parameter randomization for batched experiments."""

import math
from dataclasses import replace

import jax


def randomize(
    model, key, *, mass_scale=(0.8, 1.2), damping_scale=(0.8, 1.2), friction_scale=(0.8, 1.2)
):
    """Scale each link's mass and inertia together, preserving realizability.

    Bounds are positive static configuration. Geometry, dt and actuator limits
    stay fixed. Use vmap over explicit keys to generate independent models.
    """
    for bounds in (mass_scale, damping_scale, friction_scale):
        if (
            len(bounds) != 2
            or not all(math.isfinite(x) for x in bounds)
            or not 0 < bounds[0] <= bounds[1]
        ):
            raise ValueError("randomization ranges must be positive, finite and ordered")
    mass_key, damping_key, friction_key = jax.random.split(key, 3)
    mass_factor = jax.random.uniform(
        mass_key, model.mass.shape, minval=mass_scale[0], maxval=mass_scale[1]
    )
    damping_factor = jax.random.uniform(
        damping_key, model.damping.shape, minval=damping_scale[0], maxval=damping_scale[1]
    )
    friction_factor = jax.random.uniform(
        friction_key, (), minval=friction_scale[0], maxval=friction_scale[1]
    )
    return replace(
        model,
        mass=model.mass * mass_factor,
        inertia=model.inertia * mass_factor[:, None, None],
        damping=model.damping * damping_factor,
        friction=model.friction * friction_factor,
    )
