"""Planar fixed-base serial-chain dynamics, in SI units.

q contains relative joint angles; zero points along +x. Gravity points -y.
All runtime objects are immutable PyTrees, with one scalar per revolute joint.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Model(NamedTuple):
    length: jax.Array
    mass: jax.Array
    inertia: jax.Array
    damping: jax.Array
    torque_limit: jax.Array
    gravity: jax.Array
    dt: jax.Array


class State(NamedTuple):
    q: jax.Array
    qd: jax.Array


def make_model(
    lengths=(0.7, 0.5), masses=(1.0, 0.7), *, damping=0.1, torque_limit=10.0, gravity=9.81, dt=0.002
):
    """Build uniform rods with center-of-mass inertia m L² / 12.

    This validated host-side constructor is used outside jit.
    """
    if len(lengths) == 0 or len(lengths) != len(masses):
        raise ValueError("lengths and masses must have equal nonzero lengths")
    if any(not math.isfinite(x) or x <= 0 for x in (*lengths, *masses)):
        raise ValueError("lengths and masses must be finite and positive")
    if any(not math.isfinite(x) for x in (damping, torque_limit, gravity, dt)):
        raise ValueError("parameters must be finite")
    if dt <= 0 or damping < 0 or torque_limit <= 0 or gravity < 0:
        raise ValueError("require dt/torque_limit > 0 and damping/gravity >= 0")
    length, mass = jnp.asarray(lengths, dtype=float), jnp.asarray(masses, dtype=float)
    return Model(
        length,
        mass,
        mass * length**2 / 12,
        jnp.full_like(length, damping),
        jnp.full_like(length, torque_limit),
        jnp.asarray(gravity),
        jnp.asarray(dt),
    )


def kinematics(model, q):
    """Return base and joint endpoints, shape (n+1, 2)."""
    angle = jnp.cumsum(q)
    segments = model.length[:, None] * jnp.stack((jnp.cos(angle), jnp.sin(angle)), -1)
    return jnp.concatenate((jnp.zeros((1, 2), dtype=q.dtype), jnp.cumsum(segments, 0)))


def centers(model, q):
    points = kinematics(model, q)
    return (points[:-1] + points[1:]) / 2


def mass_matrix(model, q):
    jac = jax.jacfwd(centers, argnums=1)(model, q)
    angular_jac = jnp.tril(jnp.ones((q.size, q.size), dtype=q.dtype))
    return jnp.einsum("lci,l,lcj->ij", jac, model.mass, jac) + angular_jac.T @ (
        model.inertia[:, None] * angular_jac
    )


def potential_energy(model, q):
    return jnp.sum(model.mass * model.gravity * centers(model, q)[:, 1])


def energy(model, state):
    return 0.5 * state.qd @ mass_matrix(model, state.q) @ state.qd + potential_energy(
        model, state.q
    )


def acceleration(model, state, torque):
    """M qdd = tau - C(q, qd) - grad(V) - damping*qd.

    C = (dM/dt) qd - 1/2 grad_q(qd.T M qd), from Euler-Lagrange.
    """
    q, qd = state
    matrix, matrix_dot = jax.jvp(lambda x: mass_matrix(model, x), (q,), (qd,))
    kinetic_grad = jax.grad(lambda x: 0.5 * qd @ mass_matrix(model, x) @ qd)(q)
    coriolis = matrix_dot @ qd - kinetic_grad
    gravity_force = jax.grad(potential_energy, argnums=1)(model, q)
    applied = jnp.clip(torque, -model.torque_limit, model.torque_limit)
    return jnp.linalg.solve(matrix, applied - coriolis - gravity_force - model.damping * qd)


def step(model, state, torque):
    """One semi-implicit Euler step; torque is in N m."""
    qd = state.qd + model.dt * acceleration(model, state, torque)
    return State(state.q + model.dt * qd, qd)


def rollout(model, state, torques):
    """Return final state and post-step states for torques shaped (time, joints)."""

    def advance(current, torque):
        following = step(model, current, torque)
        return following, following

    return jax.lax.scan(advance, state, torques)
