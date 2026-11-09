"""Native 3D RL collection with deterministic keys and physical randomization.

Run from the installed checkout: python examples/batched_3d.py
This collects random-policy transitions; it does not train an RL policy.
"""

import argparse
import time

import jax
import jax.numpy as jnp

from robogogogo import env3d, models
from robogogogo.randomization import randomize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1:
        parser.error("batch and steps must be positive")
    model = models.arm6()
    model_key, reset_key, rollout_key = jax.random.split(jax.random.key(42), 3)
    batch_models = jax.vmap(lambda key: randomize(model, key))(
        jax.random.split(model_key, args.batch)
    )
    states, _ = jax.vmap(env3d.reset)(batch_models, jax.random.split(reset_key, args.batch))

    @jax.jit
    def collect(states, key):
        def tick(carry, _):
            states, key = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            actions = jax.random.uniform(action_key, (args.batch, 6), minval=-0.1, maxval=0.1)
            transitions = jax.vmap(env3d.step)(batch_models, states, actions)
            fresh, _ = jax.vmap(env3d.reset)(batch_models, jax.random.split(reset_key, args.batch))
            done = transitions.terminated | transitions.truncated
            following = jax.tree.map(
                lambda new, old: jnp.where(
                    done.reshape((args.batch,) + (1,) * (old.ndim - 1)), new, old
                ),
                fresh,
                transitions.state,
            )
            return (following, key), transitions

        return jax.lax.scan(tick, (states, key), None, length=args.steps)

    start = time.perf_counter()
    _, transitions = collect(states, rollout_key)
    transitions.reward.block_until_ready()
    first = time.perf_counter() - start
    start = time.perf_counter()
    _, transitions = collect(states, rollout_key)
    transitions.reward.block_until_ready()
    runtime = time.perf_counter() - start
    print(f"JAX {jax.__version__} / {jax.default_backend()} / {transitions.reward.dtype}")
    print(
        f"Observations: {transitions.observation.shape}; first call (includes compilation): {first:.3f} s"
    )
    print(
        f"Warm rollout: {runtime:.3f} s; {args.batch * args.steps / runtime:.0f} control transitions/s"
    )
    print(
        f"Mean reward: {float(jnp.mean(transitions.reward)):.5f}; numerical terminations: {int(jnp.sum(transitions.terminated))}"
    )


if __name__ == "__main__":
    main()
