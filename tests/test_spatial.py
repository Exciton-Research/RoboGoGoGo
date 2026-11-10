from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fixtures import env3d, models

from robogogogo import control
from robogogogo import spatial as sp


def mixed_tree():
    b = sp.ModelBuilder()
    a = b.add_link(
        "root",
        axis=(0, 1, 0),
        origin=(0.1, 0, 0.5),
        mass=1.2,
        com=(0.2, 0.04, 0),
        inertia=np.diag([0.02, 0.03, 0.04]),
    )
    b.add_link(
        "slide",
        parent=a,
        joint="prismatic",
        axis=(1, 0, 0),
        rpy=(0.2, 0.4, -0.1),
        origin=(0.4, 0, 0),
        mass=0.6,
        com=(0.1, 0.03, 0),
        inertia=np.diag([0.01, 0.02, 0.02]),
    )
    c = b.add_link(
        "branch",
        parent=a,
        axis=(0, 0, 1),
        origin=(0.1, 0.2, 0),
        mass=0.8,
        com=(0.1, 0.1, 0),
        inertia=np.diag([0.02, 0.03, 0.04]),
    )
    b.add_link(
        "payload",
        parent=c,
        joint="fixed",
        origin=(0.2, 0, 0),
        rpy=(0.1, 0.2, 0.3),
        mass=0.3,
        com=(0.02, 0, 0),
        inertia=np.eye(3) * 0.005,
    )
    return b.build()


def independent_kinetic_energy(model, q, qd):
    # Independent Cartesian kinetic-energy construction, not inverse dynamics.
    def poses(q):
        pose = sp.forward_kinematics(model, q)
        return (pose.position + jnp.einsum("bij,bj->bi", pose.rotation, model.com), pose.rotation)

    (_, rotation), (velocity, rotation_dot) = jax.jvp(poses, (q,), (qd,))
    omega_cross = jnp.einsum("bji,bjk->bik", rotation, rotation_dot)
    omega = jnp.stack((omega_cross[:, 2, 1], omega_cross[:, 0, 2], omega_cross[:, 1, 0]), axis=1)
    return 0.5 * jnp.sum(model.mass * jnp.sum(velocity**2, axis=1)) + 0.5 * jnp.einsum(
        "bi,bij,bj->", omega, model.inertia, omega
    )


def test_newton_euler_matches_independent_lagrange_tree():
    m = mixed_tree()
    q, v, a = jnp.array([0.4, 0.2, -0.3]), jnp.array([0.7, -0.2, 0.4]), jnp.array([-0.3, 0.1, 0.2])
    momentum = jax.grad(independent_kinetic_energy, argnums=2)
    mass = jax.jacfwd(momentum, argnums=2)(m, q, v)
    momentum_q = jax.jacfwd(momentum, argnums=1)(m, q, v)
    force = mass @ a + momentum_q @ v - jax.grad(independent_kinetic_energy, argnums=1)(m, q, v)
    force += jax.grad(sp.potential_energy, argnums=1)(m, q)
    np.testing.assert_allclose(sp.inverse_dynamics(m, q, v, a), force, atol=1e-10)
    np.testing.assert_allclose(sp.mass_matrix(m, q), mass, atol=1e-10)
    assert np.linalg.eigvalsh(np.asarray(mass)).min() > 0


def test_analytic_3d_pendulum():
    b = sp.ModelBuilder()
    b.add_link(
        "rod",
        axis=(0, 1, 0),
        mass=3.0,
        com=(1, 0, 0),
        inertia=np.diag([0.01, 1.0, 1.0]),
        damping=0.0,
    )
    m = b.build()
    q = jnp.array([0.3])
    np.testing.assert_allclose(sp.mass_matrix(m, q), [[4.0]], atol=1e-12)
    # Rotation about +y sends +x toward -z.
    np.testing.assert_allclose(
        sp.acceleration(m, sp.State(q, jnp.zeros(1)), jnp.array([1.2])),
        [(1.2 + 3 * 9.81 * np.cos(0.3)) / 4],
        atol=1e-12,
    )


def test_gravity_is_potential_gradient_and_mass_symmetric():
    m = models.arm6()
    zero = jnp.zeros(6)
    np.testing.assert_allclose(
        sp.inverse_dynamics(m, m.home, zero, zero),
        jax.grad(sp.potential_energy, argnums=1)(m, m.home),
        atol=1e-10,
    )
    matrix = sp.mass_matrix(m, m.home)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(matrix)).min() > 0


def test_contact_settles_at_compliant_equilibrium():
    m = models.slider()
    final, _ = jax.jit(sp.rollout)(m, sp.initial_state(m), jnp.zeros((2500, 1)))
    expected = 0.1 - 9.81 / float(m.contact_stiffness)
    np.testing.assert_allclose(final.q, [expected], atol=2e-5)
    assert abs(float(final.qd[0])) < 1e-4
    contacts, generalized = sp.plane_contacts(m, final)
    np.testing.assert_allclose(generalized, [9.81], atol=2e-3)
    assert float(contacts.force[0, 2]) >= 0
    airborne, force = sp.plane_contacts(m, sp.initial_state(m))
    np.testing.assert_array_equal(force, [0.0])


def test_surface_friction_has_rotational_torque_and_coulomb_bound():
    b = sp.ModelBuilder(gravity=(0, 0, 0))
    body = b.add_link(
        "wheel",
        axis=(0, 1, 0),
        origin=(0, 0, 0.09),
        mass=1.0,
        inertia=np.eye(3) * 0.004,
        damping=0.0,
    )
    b.add_sphere(body, (0, 0, 0), 0.1)
    m = b.build()
    contacts, force = sp.plane_contacts(m, sp.State(jnp.zeros(1), jnp.ones(1)))
    assert float(force[0]) < 0  # friction opposes rotation
    assert float(jnp.linalg.norm(contacts.force[0, :2])) <= float(m.friction * contacts.force[0, 2])


def test_joint_stops_and_actuator_saturation():
    m = models.slider()
    m = replace(
        m,
        lower=jnp.array([0.2]),
        upper=jnp.array([0.8]),
        sphere_body=jnp.array([], dtype=int),
        sphere_center=jnp.zeros((0, 3)),
        sphere_radius=jnp.zeros(0),
    )
    assert float(sp.passive_forces(m, sp.State(jnp.array([0.1]), jnp.array([-1.0])))[0]) > 0
    assert float(sp.passive_forces(m, sp.State(jnp.array([0.9]), jnp.array([1.0])))[0]) < 0
    state = sp.initial_state(m)
    np.testing.assert_allclose(
        sp.step(m, state, jnp.array([1000.0])).q, sp.step(m, state, m.effort).q
    )


def test_jit_vmap_rollout_gradient():
    m = mixed_tree()
    state = sp.State(jnp.array([0.3, 0.1, -0.2]), jnp.zeros(3))

    def objective(torque):
        final, _ = sp.rollout(m, state, jnp.tile(torque, (8, 1)))
        return jnp.sum(final.q**2)

    torque = jnp.array([0.3, -0.2, 0.1])
    gradient = jax.jit(jax.grad(objective))(torque)
    eps = 1e-4
    finite = jnp.array(
        [
            (objective(torque + eps * axis) - objective(torque - eps * axis)) / (2 * eps)
            for axis in jnp.eye(3)
        ]
    )
    np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-5)
    batch = jax.jit(jax.vmap(lambda t: sp.step(m, state, t)))(jnp.stack((torque, torque)))
    np.testing.assert_allclose(batch.q[0], sp.step(m, state, torque).q, atol=1e-12)


def test_energy_converges_with_timestep():
    b = sp.ModelBuilder(dt=0.002)
    b.add_link(
        "pendulum",
        axis=(0, 1, 0),
        mass=1.0,
        com=(0.5, 0, 0),
        inertia=np.diag([0.001, 1 / 12, 1 / 12]),
        damping=0.0,
    )
    m = b.build()
    state = sp.State(jnp.array([0.3]), jnp.zeros(1))

    def drift(model, count):
        _, history = jax.jit(sp.rollout)(model, state, jnp.zeros((count, 1)))
        energy = jax.vmap(lambda s: sp.energy(model, s))(history)
        return float(jnp.max(jnp.abs(energy - sp.energy(model, state))))

    coarse = drift(m, 500)
    fine = drift(replace(m, dt=m.dt / 2), 1000)
    assert fine < 0.6 * coarse
    assert coarse < 0.05


def test_ik_and_computed_torque_controller():
    m = models.arm6()
    target = sp.point_position(m, m.home + jnp.array([0.3, 0.1, -0.2, 0.2, 0.1, 0.0]))
    desired, error = jax.jit(control.inverse_kinematics)(m, target)
    assert float(error) < 1e-4

    def advance(state, _):
        torque = control.computed_torque(m, state, desired)
        return sp.step(m, state, torque, substeps=5), None

    final, _ = jax.jit(lambda s: jax.lax.scan(advance, s, None, length=300))(sp.initial_state(m))
    assert float(jnp.linalg.norm(sp.point_position(m, final.q) - target)) < 0.001


def test_env3d_reproducible_batched_horizon_and_failure():
    m = models.slider()
    state, observation = env3d.reset(m, jax.random.key(2))
    same, _ = env3d.reset(m, jax.random.key(2))
    np.testing.assert_array_equal(state.physics.q, same.physics.q)
    assert observation.shape == (8,)
    states, _ = jax.vmap(lambda key: env3d.reset(m, key))(jax.random.split(jax.random.key(1), 4))
    result = jax.jit(jax.vmap(lambda s: env3d.step(m, s, jnp.zeros(1), horizon=1)))(states)
    assert bool(jnp.all(result.truncated)) and not bool(jnp.any(result.terminated))
    failed = env3d.step(m, state, jnp.array([jnp.nan]))
    assert bool(failed.terminated)
    assert bool(jnp.all(jnp.isfinite(failed.observation)))
    assert float(failed.reward) == -100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(axis=(0, 0, 0)),
        dict(mass=-1),
        dict(parent=2),
        dict(inertia=np.diag([1.0, 1.0, 4.0])),
        dict(limits=(1, -1)),
        dict(home=float("nan")),
    ],
)
def test_builder_rejects_invalid_physics(kwargs):
    b = sp.ModelBuilder()
    params = dict(inertia=np.eye(3), **{})
    params.update(kwargs)
    with pytest.raises(ValueError):
        b.add_link("bad", **params)


def test_contact_force_gradient_matches_finite_difference_away_from_switch():
    b = sp.ModelBuilder(gravity=(0, 0, 0))
    body = b.add_link(
        "sphere", axis=(0, 1, 0), origin=(0, 0, 0.08), mass=1.0, inertia=np.eye(3) * 0.004
    )
    b.add_sphere(body, (0.1, 0, 0), 0.1)
    m = b.build()

    def force(q):
        return sp.plane_contacts(m, sp.State(q, jnp.array([0.7])))[1]

    q, eps = jnp.array([0.4]), 1e-5
    gradient = jax.jacfwd(force)(q)
    finite = (force(q + eps) - force(q - eps)) / (2 * eps)
    np.testing.assert_allclose(gradient[:, 0], finite, rtol=1e-5, atol=1e-6)


def test_domain_randomization_preserves_inertia_mass_ratio():
    from robogogogo.randomization import randomize

    m = models.arm6()
    randomized = jax.jit(randomize)(m, jax.random.key(5))
    mask = m.mass > 0
    np.testing.assert_allclose(
        randomized.inertia[mask] / randomized.mass[mask, None, None],
        m.inertia[mask] / m.mass[mask, None, None],
        atol=1e-12,
    )
    assert bool(jnp.all(randomized.mass[mask] >= 0.8 * m.mass[mask]))
    assert bool(jnp.all(randomized.mass[mask] <= 1.2 * m.mass[mask]))


def test_stiff_joint_stop_on_light_inertia_remains_bounded():
    b = sp.ModelBuilder(dt=0.002, gravity=(0, 0, 0))
    b.add_link(
        "wrist", mass=0.01, inertia=np.eye(3) * 1e-5, damping=0.01, limits=(-0.1, 0.1), effort=1.0
    )
    model = b.build(limit_stiffness=500.0, limit_damping=20.0)
    initial = sp.State(jnp.array([0.09]), jnp.array([10.0]))
    _, history = jax.jit(sp.rollout)(model, initial, jnp.ones((1000, 1)) * 0.1)
    assert bool(jnp.all(jnp.isfinite(history.q)))
    assert float(jnp.max(jnp.abs(history.q))) < 0.12


def test_step_rejects_broadcasted_torque_shape():
    model = models.arm6()
    with pytest.raises(ValueError, match="shape"):
        sp.step(model, sp.initial_state(model), jnp.array([1.0]))
