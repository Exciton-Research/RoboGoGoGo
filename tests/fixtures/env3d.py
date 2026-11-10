"""Functional, batched-friendly 3D reaching environment for robotics research."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from robogogogo import spatial as sp


class EnvState(NamedTuple):
    physics: sp.State
    target: jax.Array
    elapsed: jax.Array


class Transition(NamedTuple):
    state: EnvState
    observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    distance: jax.Array
    success: jax.Array


def observation(model, state):
    """[q, qd, target_xyz, target_minus_tip_xyz], shape 2*n+6."""
    tip = sp.point_position(model, state.physics.q)
    return jnp.concatenate((state.physics.q, state.physics.qd, state.target, state.target - tip))


def reset(model, key):
    start, goal = jax.random.split(key)
    # Local reachable workspace around home; not collision-free certified.
    low = jnp.maximum(model.home - 0.5, model.lower)
    high = jnp.minimum(model.home + 0.5, model.upper)
    q = jax.random.uniform(start, model.home.shape, minval=low, maxval=high)
    target_q = jax.random.uniform(goal, model.home.shape, minval=low, maxval=high)
    state = EnvState(
        sp.State(q, jnp.zeros_like(q)), sp.point_position(model, target_q), jnp.array(0)
    )
    return state, observation(model, state)


def step(model, state, action, *, substeps=5, horizon=1000):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    action = jnp.clip(action, -1.0, 1.0)
    physics = sp.step(model, state.physics, action * model.effort, substeps=substeps)
    finite = jnp.all(jnp.isfinite(physics.q)) & jnp.all(jnp.isfinite(physics.qd))
    # Failure transitions remain finite, retain the previous valid state, and
    # terminate with a penalty. They must not be interpreted as successful steps.
    physics = jax.tree.map(lambda new, old: jnp.where(finite, new, old), physics, state.physics)
    following = EnvState(physics, state.target, state.elapsed + 1)
    error = sp.point_position(model, physics.q) - state.target
    squared_distance = jnp.sum(error**2)
    reward = jnp.where(finite, -squared_distance - 0.001 * jnp.sum(action**2), -100.0)
    return Transition(
        following,
        observation(model, following),
        reward,
        ~finite,
        following.elapsed >= horizon,
        jnp.sqrt(squared_distance),
        finite & (squared_distance < 0.02**2),
    )
