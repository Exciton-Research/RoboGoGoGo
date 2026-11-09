"""Native JAX simulation contacts; right-handed, z-up, SI units."""

import jax
import jax.numpy as jnp

from .kinematics import forward_kinematics, sphere_positions
from .model import Contacts


def plane_contacts(model, state):
    """Experimental compliant sphere contacts against the static z=0 plane.

    Normal force is unilateral spring-damper; tangential friction uses a smooth
    velocity law bounded by mu*normal. No sphere-sphere/self-collision is modeled.
    Tangent velocities use the sphere's surface point (including link rotation).
    """
    positions = sphere_positions(model, state.q)
    pose = forward_kinematics(model, state.q)
    # Choose the instantaneous local material point at the bottom of each sphere.
    down = jnp.array([0.0, 0.0, -1.0], dtype=state.q.dtype)
    offsets = (
        model.sphere_center
        + jnp.einsum("sji,j->si", pose.rotation[model.sphere_body], down)
        * model.sphere_radius[:, None]
    )

    def surface(q):
        poses = forward_kinematics(model, q)
        return poses.position[model.sphere_body] + jnp.einsum(
            "sij,sj->si", poses.rotation[model.sphere_body], offsets
        )

    jac = jax.jacfwd(surface)(state.q)
    velocity = jnp.einsum("sij,j->si", jac, state.qd)
    penetration = model.sphere_radius - positions[:, 2]
    normal = jnp.where(
        penetration > 0,
        jnp.maximum(
            0.0, model.contact_stiffness * penetration - model.contact_damping * velocity[:, 2]
        ),
        0.0,
    )
    tangent = velocity[:, :2]
    # Regularized Coulomb friction, with characteristic slip speed 0.01 m/s.
    friction = (
        -model.friction
        * normal[:, None]
        * tangent
        / jnp.sqrt(jnp.sum(tangent**2, axis=1, keepdims=True) + 0.01**2)
    )
    forces = jnp.concatenate((friction, normal[:, None]), axis=1)
    return Contacts(
        positions + down * model.sphere_radius[:, None], jnp.maximum(penetration, 0.0), forces
    ), jnp.einsum("sij,si->j", jac, forces)
