"""Run: python examples/reach.py [--headless]. No learned policy required."""

import argparse

import jax
import jax.numpy as jnp

from robogogogo import env, kinematics, make_model
from robogogogo.physics import potential_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model = make_model(torque_limit=20.0)
    state, _ = env.reset(model, jax.random.key(7))
    # A known reachable joint configuration gives a deterministic PD baseline.
    desired_q = jnp.array([-0.6, 1.4])
    state = state._replace(target=kinematics(model, desired_q)[-1])

    @jax.jit
    def advance(state):
        q, qd = state.physics
        error = jnp.arctan2(jnp.sin(desired_q - q), jnp.cos(desired_q - q))
        gravity = jax.grad(potential_energy, argnums=1)(model, q)
        torque = 12.0 * error - 3.0 * qd + gravity
        return env.step(model, state, torque / model.torque_limit)

    initial_error = float(jnp.linalg.norm(kinematics(model, state.physics.q)[-1] - state.target))
    frames = [kinematics(model, state.physics.q)]
    for _ in range(500):
        result = advance(state)
        state = result.state
        frames.append(kinematics(model, state.physics.q))
        if bool(result.terminated | result.truncated):
            break
    final_error = float(jnp.linalg.norm(frames[-1][-1] - state.target))
    print(f"Tip error: {initial_error:.4f} m -> {final_error:.4f} m")
    print(f"Control ticks: {int(state.elapsed)}; final reward: {float(result.reward):.5f}")
    if args.headless:
        return
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    reach = float(jnp.sum(model.length)) * 1.15
    ax.set(
        xlim=(-reach, reach),
        ylim=(-reach, reach),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="robogogogo • torque-controlled planar arm",
    )
    ax.grid(alpha=0.2)
    ax.plot(*state.target, "rx", markersize=12, label="Target")
    (line,) = ax.plot([], [], "o-", linewidth=5, label="Robot")
    ax.legend()

    def draw(index):
        points = frames[index]
        line.set_data(points[:, 0], points[:, 1])
        return (line,)

    _animation = FuncAnimation(fig, draw, frames=len(frames), interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
