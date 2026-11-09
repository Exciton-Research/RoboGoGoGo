import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import State, acceleration, energy, env, make_model, mass_matrix, rollout, step


def test_single_rod_matches_analytic_pendulum():
    model = make_model((2.0,), (3.0,), damping=0)
    state = State(jnp.array([0.3]), jnp.array([0.7]))
    np.testing.assert_allclose(mass_matrix(model, state.q), [[4.0]], atol=1e-12)
    expected = (1.2 - 3 * 9.81 * np.cos(0.3)) / 4
    np.testing.assert_allclose(acceleration(model, state, jnp.array([1.2])), [expected], atol=1e-12)


def test_double_rod_mass_matrix_matches_closed_form():
    model = make_model((0.7, 0.5), (1.0, 0.7))
    q = jnp.array([0.3, -0.8])
    a, b = 0.7, 0.5
    m, n = 1.0, 0.7
    c = np.cos(q[1])
    expected = [
        [m * a * a / 3 + n * (a * a + b * b / 3 + a * b * c), n * (b * b / 3 + a * b * c / 2)],
        [n * (b * b / 3 + a * b * c / 2), n * b * b / 3],
    ]
    np.testing.assert_allclose(mass_matrix(model, q), expected, atol=1e-12)


def test_unforced_motion_energy_error_is_bounded():
    model = make_model(damping=0, dt=0.0005)
    state = State(jnp.array([0.4, -0.7]), jnp.zeros(2))

    def drift(model, steps):
        _, states = jax.jit(rollout)(model, state, jnp.zeros((steps, 2)))
        energies = jax.vmap(lambda s: energy(model, s))(states)
        return float(jnp.max(jnp.abs(energies - energy(model, state))))

    coarse = drift(model, 2000)
    fine = drift(model._replace(dt=model.dt / 2), 4000)
    # First-order integration should converge as dt decreases over the same 1 s.
    gravitational_scale = float(jnp.sum(model.mass) * model.gravity * jnp.sum(model.length))
    assert coarse / gravitational_scale < 0.005
    assert fine < 0.6 * coarse


def test_jit_batching_and_gradient_match_finite_difference():
    model = make_model()
    state = State(jnp.array([0.2, -0.4]), jnp.zeros(2))

    def objective(torque):
        final, _ = rollout(model, state, jnp.tile(torque, (20, 1)))
        return jnp.sum(final.q**2)

    torque = jnp.array([0.3, -0.1])
    analytic = jax.jit(jax.grad(objective))(torque)
    eps = 1e-5
    finite = jnp.array(
        [
            (objective(torque + eps * axis) - objective(torque - eps * axis)) / (2 * eps)
            for axis in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(analytic, finite, rtol=1e-5, atol=1e-8)
    batched = jax.jit(jax.vmap(lambda t: step(model, state, t)))(jnp.stack([torque, torque]))
    np.testing.assert_allclose(batched.q[0], step(model, state, torque).q)


def test_torque_limits():
    model = make_model(torque_limit=2.0)
    state = State(jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_allclose(
        step(model, state, jnp.full(2, 100)).q, step(model, state, jnp.full(2, 2.0)).q
    )


def test_environment_reset_batch_and_time_limit():
    model = make_model()
    state, obs = env.reset(model, jax.random.key(4))
    same, _ = env.reset(model, jax.random.key(4))
    np.testing.assert_array_equal(state.physics.q, same.physics.q)
    assert obs.shape == (10,)
    states, observations = jax.jit(jax.vmap(lambda key: env.reset(model, key)))(
        jax.random.split(jax.random.key(0), 8)
    )
    result = jax.jit(jax.vmap(lambda s: env.step(model, s, jnp.zeros(2), horizon=1)))(states)
    assert observations.shape == (8, 10)
    assert bool(jnp.all(result.truncated))
    assert not bool(jnp.any(result.terminated))
    assert bool(jnp.all(jnp.isfinite(result.reward)))


@pytest.mark.parametrize(
    "kwargs", [dict(lengths=()), dict(dt=0), dict(masses=(-1.0, 1.0)), dict(gravity=float("nan"))]
)
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)
