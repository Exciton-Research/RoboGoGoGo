"""Functional reaching environment. Actions are normalized joint torques."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import Model, State, kinematics
from .physics import step as physics_step


class EnvState(NamedTuple):
    physics: State
    target: jax.Array
    elapsed: jax.Array


class Transition(NamedTuple):
    state: EnvState
    observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


def observation(model, state):
    tip = kinematics(model, state.physics.q)[-1]
    return jnp.concatenate(
        (
            jnp.cos(state.physics.q),
            jnp.sin(state.physics.q),
            state.physics.qd,
            state.target,
            state.target - tip,
        )
    )


def reset(model: Model, key):
    """Sample angles and a reachable target using explicit PRNG state."""
    start_key, target_key = jax.random.split(key)
    q = jax.random.uniform(start_key, model.length.shape, minval=-jnp.pi, maxval=jnp.pi)
    target_q = jax.random.uniform(target_key, model.length.shape, minval=-jnp.pi, maxval=jnp.pi)
    state = EnvState(State(q, jnp.zeros_like(q)), kinematics(model, target_q)[-1], jnp.array(0))
    return state, observation(model, state)


def step(model, state, action, *, substeps=10, horizon=500):
    """Advance one control tick; substeps/horizon are positive static integers.

    Reward = negative squared tip error minus 0.01 * squared action.
    Episodes end at horizon; callers must reset after truncation/termination.
    No implicit reset is performed, so terminal observations are preserved.
    """
    if substeps < 1 or horizon < 1:
        raise ValueError("substeps and horizon must be positive")
    action = jnp.clip(action, -1.0, 1.0)

    def advance(physics, _):
        return physics_step(model, physics, action * model.torque_limit), None

    physics, _ = jax.lax.scan(advance, state.physics, None, length=substeps)
    following = EnvState(physics, state.target, state.elapsed + 1)
    error = kinematics(model, physics.q)[-1] - state.target
    reward = -jnp.sum(error**2) - 0.01 * jnp.sum(action**2)
    finite = jnp.all(jnp.isfinite(physics.q)) & jnp.all(jnp.isfinite(physics.qd))
    return Transition(
        following, observation(model, following), reward, ~finite, following.elapsed >= horizon
    )
