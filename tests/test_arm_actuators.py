"""Actuator envelope invariants, including braking and passive dissipation."""

import jax
import jax.numpy as jnp
import numpy as np

from robogogogo.robot.actuators import delivered_torque, gearbox_friction


def test_torque_speed_droop_preserves_braking_and_symmetry():
    speed = jnp.array([0.0, 1.5, 3.0, 4.0, -3.0])
    drive = jax.jit(delivered_torque)
    output = drive(jnp.full(5, 100.0), speed, jnp.full(5, 60.0), jnp.full(5, 3.0))
    np.testing.assert_allclose(output, [60, 30, 0, 0, 60])
    np.testing.assert_allclose(
        drive(jnp.full(5, -100.0), -speed, jnp.full(5, 60.0), jnp.full(5, 3.0)), -output
    )
    np.testing.assert_allclose(
        drive(jnp.full(5, 10.0), speed, jnp.full(5, 60.0), jnp.full(5, jnp.inf)), 10
    )


def test_friction_never_adds_energy_and_has_no_artificial_rest_force():
    speed = jnp.linspace(-5, 5, 101)
    force = jax.jit(gearbox_friction)(speed, jnp.full_like(speed, 0.12))
    assert np.all(np.asarray(force * speed) <= 0)
    assert np.max(np.abs(force)) <= 0.12 + 1e-7
    assert float(gearbox_friction(jnp.array(0.0), jnp.array(0.12))) == 0
    np.testing.assert_allclose(gearbox_friction(-speed, 0.12), -force)
