"""Native JAX simulation integration; right-handed, z-up, SI units."""

from functools import partial

import jax
import jax.numpy as jnp

from .dynamics import acceleration
from .model import State


@partial(jax.jit, static_argnames=("substeps",))
def step(model, state, torque, *, substeps=1):
    """Semi-implicit positions with implicit damping/stops; substep = model.dt."""
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive static integer")
    torque = jnp.asarray(torque, dtype=state.q.dtype)
    if (
        state.q.shape != model.home.shape
        or state.qd.shape != model.home.shape
        or torque.shape != model.home.shape
    ):
        raise ValueError("q, qd and torque must each have shape (number_of_joints,)")

    def advance(current, _):
        qd = current.qd + model.dt * acceleration(model, current, torque, implicit_dt=model.dt)
        return State(current.q + model.dt * qd, qd), None

    return jax.lax.scan(advance, state, None, length=substeps)[0]


def rollout(model, state, torques, *, substeps=1):
    def advance(current, torque):
        following = step(model, current, torque, substeps=substeps)
        return following, following

    return jax.lax.scan(advance, state, torques)


def initial_state(model):
    return State(model.home, jnp.zeros_like(model.home))
