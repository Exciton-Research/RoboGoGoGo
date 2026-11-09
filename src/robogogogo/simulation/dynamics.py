"""Native JAX simulation dynamics; right-handed, z-up, SI units."""

import jax
import jax.numpy as jnp

from .contacts import plane_contacts
from .kinematics import forward_kinematics
from .math import rotation


def inverse_dynamics(model, q, qd, qdd, *, gravity=None):
    """Recursive Newton–Euler: M(q) qdd + C(q,qd) + G(q).

    Excludes passive damping, joint stops, contacts and actuator clipping.
    Fixed descendants contribute mass and wrench to their ancestors.
    """
    gravity = model.gravity if gravity is None else gravity
    positions, rotations, velocities, alphas, accelerations, axes = [], [], [], [], [], []
    forces, moments = [], []
    zero = jnp.zeros(3, dtype=q.dtype)
    for i, (parent, kind, index) in enumerate(zip(model.parents, model.kinds, model.indices)):
        p0 = zero if parent == -1 else positions[parent]
        r0 = jnp.eye(3, dtype=q.dtype) if parent == -1 else rotations[parent]
        w0 = zero if parent == -1 else velocities[parent]
        alpha0 = zero if parent == -1 else alphas[parent]
        a0 = zero if parent == -1 else accelerations[parent]
        r = r0 @ model.orientation[i]
        axis = r @ model.axis[i]
        offset = r0 @ model.origin[i]
        if kind == "prismatic":
            offset = offset + axis * q[index]
        p = p0 + offset
        w, alpha = w0, alpha0
        a = a0 + jnp.cross(alpha0, offset) + jnp.cross(w0, jnp.cross(w0, offset))
        if kind in ("revolute", "continuous"):
            w = w0 + axis * qd[index]
            alpha = alpha0 + axis * qdd[index] + jnp.cross(w0, axis * qd[index])
            r = r @ rotation(model.axis[i], q[index])
        elif kind == "prismatic":
            a = a + axis * qdd[index] + 2 * jnp.cross(w0, axis * qd[index])
        c = r @ model.com[i]
        inertia = r @ model.inertia[i] @ r.T
        force = model.mass[i] * (a + jnp.cross(alpha, c) + jnp.cross(w, jnp.cross(w, c)) - gravity)
        moment = inertia @ alpha + jnp.cross(w, inertia @ w) + jnp.cross(c, force)
        positions.append(p)
        rotations.append(r)
        velocities.append(w)
        alphas.append(alpha)
        accelerations.append(a)
        axes.append(axis)
        forces.append(force)
        moments.append(moment)
    torque = jnp.zeros_like(q)
    for i in reversed(range(len(model.parents))):
        index, parent, kind = model.indices[i], model.parents[i], model.kinds[i]
        if index >= 0:
            wrench = forces[i] if kind == "prismatic" else moments[i]
            torque = torque.at[index].set(jnp.dot(axes[i], wrench))
        if parent >= 0:
            forces[parent] = forces[parent] + forces[i]
            moments[parent] = (
                moments[parent]
                + moments[i]
                + jnp.cross(positions[i] - positions[parent], forces[i])
            )
    return torque


def mass_matrix(model, q):
    zero = jnp.zeros_like(q)
    return jax.vmap(
        lambda a: inverse_dynamics(model, q, zero, a, gravity=jnp.zeros(3)), out_axes=1
    )(jnp.eye(q.size, dtype=q.dtype))


def potential_energy(model, q):
    pose = forward_kinematics(model, q)
    centers = pose.position + jnp.einsum("bij,bj->bi", pose.rotation, model.com)
    return -jnp.sum(model.mass * (centers @ model.gravity))


def energy(model, state):
    return 0.5 * state.qd @ mass_matrix(model, state.q) @ state.qd + potential_energy(
        model, state.q
    )


def passive_forces(model, state):
    lower = jnp.maximum(model.lower - state.q, 0.0)
    upper = jnp.maximum(state.q - model.upper, 0.0)
    stop = model.limit_stiffness * (lower - upper)
    stop = stop + jnp.where(lower > 0, -model.limit_damping * jnp.minimum(state.qd, 0.0), 0.0)
    stop = stop + jnp.where(upper > 0, -model.limit_damping * jnp.maximum(state.qd, 0.0), 0.0)
    return stop - model.damping * state.qd


def acceleration(model, state, torque, *, implicit_dt=None):
    """Forward dynamics; optional implicit damping/stop linearization for stepping."""
    bias = inverse_dynamics(model, state.q, state.qd, jnp.zeros_like(state.q))
    matrix = mass_matrix(model, state.q)
    force = jnp.clip(torque, -model.effort, model.effort)
    if implicit_dt is None:
        force = force + passive_forces(model, state)
    else:
        # Solve velocity damping and locally active joint springs implicitly.
        # The active set uses the ballistic next q; contact geometry is explicit.
        h = implicit_dt
        predicted = state.q + h * state.qd
        low, high = predicted < model.lower, predicted > model.upper
        stiffness = model.limit_stiffness * (low | high)
        damping = model.damping + model.limit_damping * (
            (low & (state.qd < 0)) | (high & (state.qd > 0))
        )
        distance = jnp.where(low, model.lower - predicted, 0.0) + jnp.where(
            high, model.upper - predicted, 0.0
        )
        force = force + model.limit_stiffness * distance - damping * state.qd
        matrix = matrix + jnp.diag(h * damping + h * h * stiffness)
    if model.sphere_body.shape[0]:
        _, contact_force = plane_contacts(model, state)
        force = force + contact_force
    return jnp.linalg.solve(matrix, force - bias)
