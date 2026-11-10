"""Physical invariants and guard boundaries for the reduced-order mobile model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.robot.mobile_design import mobile_arm_definitions, mobile_dual_arm_urdf, mobile_home
from robogogogo.session import Session
from robogogogo.simulation.mobile import MobileDynamics
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    model = loads(mobile_dual_arm_urdf()).model
    return replace(model, home=jnp.asarray(mobile_home(model)))


@pytest.fixture(scope="module")
def mobile(model):
    return MobileDynamics(model, 5)


def test_tires_obey_friction_circle_load_balance_and_dissipate_slip(model, mobile):
    state = sp.initial_state(model)
    state = state._replace(qd=state.qd.at[mobile.wheels].set(5).at[mobile.base[1]].set(0.2))
    force, diagnostics = mobile.tire_forces(state, 0.3, jnp.zeros(3))
    _, loads, _, tires, slip, side = diagnostics
    assert float(jnp.sum(loads)) == pytest.approx(float(mobile.mass) * 9.81, rel=1e-5)
    assert np.all(np.linalg.norm(tires, axis=1) <= 0.3 * np.asarray(loads) + 1e-5)
    assert np.all(-np.asarray(tires[:, 0]) * slip + tires[:, 1] * side <= 0)
    assert float(force[mobile.base[0]]) > 0
    assert np.all(np.asarray(force[mobile.wheels]) < 0)
    zero, _ = mobile.tire_forces(state, 0, jnp.zeros(3))
    np.testing.assert_array_equal(zero, 0)


def test_steering_turning_reversal_and_alignment(model, mobile):
    state = sp.initial_state(model)
    angles, speeds = mobile.steering_targets(state, jnp.array([-0.2, 0, 0]))
    np.testing.assert_allclose(angles, 0, atol=1e-6)
    assert np.all(np.asarray(speeds) < 0)
    angles, speeds = mobile.steering_targets(state, jnp.array([0, 0, 0.3]))
    assert len(set(np.round(np.asarray(angles), 3))) > 1
    assert speeds[0] * speeds[1] < 0
    _, sideways = mobile.steering_targets(state, jnp.array([0, 0.2, 0]))
    np.testing.assert_allclose(sideways, 0, atol=1e-6)


def test_high_acceleration_moves_zmp_outside_support(model, mobile):
    assert float(mobile.static_margin(model.home)) > mobile.margin
    assert float(mobile.support(model.home, jnp.array([8, 0, 0]))[0]) < 0


def test_drive_and_braking_are_friction_limited(model, mobile):
    @jax.jit
    def run(mu):
        def tick(carry, i):
            state, acc = carry
            twist = jnp.where(i < 150, jnp.array([0.2, 0, 0]), jnp.zeros(3))
            next_state, acc, margin, _, _ = mobile._step(state, model.home, twist, mu, acc)
            return (next_state, acc), (next_state.q, next_state.qd, margin)

        return jax.lax.scan(tick, (sp.initial_state(model), jnp.zeros(3)), jnp.arange(300))

    (_, _), (q, qd, margins) = run(0.7)
    assert np.isfinite(q).all()
    assert float(jnp.min(margins)) > mobile.margin
    assert float(q[149, mobile.base[0]]) > 0.08
    assert abs(float(qd[-1, mobile.base[0]])) < 0.03
    (_, _), (ice_q, _, _) = run(0.0)
    assert abs(float(ice_q[-1, mobile.base[0]])) < 0.005
    assert abs(float(ice_q[-1, mobile.wheels[0]])) > 0.1


def test_session_drive_validation_lease_reset_and_no_coordinate_bypass(model):
    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    with pytest.raises(ValueError, match="Drive limits"):
        session.set_drive([0.6, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        session.set_drive([float("nan"), 0, 0])
    goal = np.asarray(model.home).copy()
    goal[0] += 0.2
    with pytest.raises(ValueError, match="drive commands"):
        session.set_joints(goal)
    session.set_drive([0.1, 0, 0], duration=0.01)
    session.advance()
    session.advance()
    assert session.elapsed >= session.drive_until
    assert session.mobile_status["guard"] == "ok"
    session.reset()
    np.testing.assert_array_equal(session.drive_twist, 0)
    assert session.drive_until == 0


def test_turn_and_torso_motion_keep_support_reserve(model, mobile):
    low = jnp.asarray(mobile_home(model, "low"))

    @jax.jit
    def run():
        def tick(carry, _):
            state, acc = carry
            following, acc, margin, _, _ = mobile._step(
                state, low, jnp.array([0.0, 0.0, 0.15]), 0.7, acc
            )
            return (following, acc), margin

        return jax.lax.scan(tick, (sp.initial_state(model), jnp.zeros(3)), None, length=500)

    (state, _), margins = run()
    assert np.isfinite(state.q).all()
    assert float(jnp.min(margins)) > mobile.margin
    assert float(state.q[mobile.base[2]]) > 0.15
    torso = model.indices[model.names.index("torso_lower")]
    assert abs(float(state.q[torso] - low[torso])) < 0.2


def test_guard_rejection_is_atomic_and_clears_drive(model):
    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    session.set_drive([0.1, 0, 0])
    before = session.state
    _, diagnostics = session.mobile.tire_forces(before, 0.7, jnp.zeros(3))
    session.mobile.step = lambda *args: (
        before._replace(q=before.q + 1),
        jnp.zeros(3),
        -0.1,
        jnp.zeros_like(before.q),
        diagnostics,
    )
    with pytest.raises(FloatingPointError, match="Stability guard"):
        session.advance()
    assert session.state is before
    assert session.elapsed == 0
    assert len(session.history) == 1
    np.testing.assert_array_equal(session.drive_twist, 0)
    assert session.mobile_status["guard"] == "blocked"
