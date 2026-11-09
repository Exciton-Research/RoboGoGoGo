"""Minimal compiled random-policy rollout across 64 independent environments."""

import jax
import jax.numpy as jnp

from robogogogo import env, make_model

model = make_model()
keys = jax.random.split(jax.random.key(0), 64)
states, _ = jax.vmap(lambda key: env.reset(model, key))(keys)


@jax.jit
def collect(states, key):
    def tick(carry, _):
        states, key = carry
        key, action_key, reset_key = jax.random.split(key, 3)
        actions = jax.random.uniform(action_key, (64, 2), minval=-1.0, maxval=1.0)
        transitions = jax.vmap(lambda s, a: env.step(model, s, a))(states, actions)
        # Preserve terminal transitions; reset only the carry for the next tick.
        reset_states, _ = jax.vmap(lambda k: env.reset(model, k))(jax.random.split(reset_key, 64))
        done = transitions.terminated | transitions.truncated
        states = jax.tree.map(
            lambda fresh, old: jnp.where(done.reshape((64,) + (1,) * (old.ndim - 1)), fresh, old),
            reset_states,
            transitions.state,
        )
        return (states, key), transitions

    return jax.lax.scan(tick, (states, key), None, length=100)


_, transitions = collect(states, jax.random.key(1))
print("Observation batch:", transitions.observation.shape)
print("Mean reward:", float(jnp.mean(transitions.reward)))
