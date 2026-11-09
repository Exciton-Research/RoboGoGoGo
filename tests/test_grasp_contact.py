"""Contact mechanics regressions: stiction, sliding, force limits and convergence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.manipulation import GraspObject
from robogogogo.simulation.grasp_contact import ContactState, contact_step, friction_force


def initial():
    return ContactState(
        jnp.array([0.2, 0.0, 0.5]),
        sp.rpy_matrix(jnp.array([0.0, 0.0, -np.pi / 2])),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.array([0.10]),
        jnp.zeros(1),
        jnp.zeros((1, 2, 6, 3)),
        jnp.zeros((9, 3)),
        jnp.array([0.10]),
    )


@jax.jit
def simulate(state, mu, force_limit, supports, dt=0.0002):
    state = state._replace(support_tangent=jnp.zeros((3 * (supports.shape[0] + 1), 3)))
    obj = GraspObject()

    def step(state, _):
        return contact_step(
            state,
            jnp.array([[0.0, 0.0, 0.5]]),
            jnp.eye(3)[None],
            jnp.zeros((1, 3)),
            jnp.zeros((1, 3)),
            jnp.array([0.015]),
            jnp.array([force_limit]),
            jnp.array([mu]),
            jnp.array([0.75 * mu]),
            obj.mass,
            jnp.asarray(obj.inertia),
            obj.radius,
            obj.half_length,
            jnp.array([0.0, 0.0, -9.81]),
            supports,
            dt,
        )

    return jax.lax.scan(step, state, None, length=2500)


def test_friction_supports_weight_but_zero_friction_slips_and_falls():
    closed, _ = simulate(initial(), 0.8, 40.0, jnp.array([[0.2, 0.0, 0.2, 0.2, 0.47]]))
    held, telemetry = simulate(closed, 0.8, 40.0, jnp.empty((0, 5)))
    dropped, low = simulate(closed, 0.0, 40.0, jnp.empty((0, 5)))
    assert abs(float(held.position[2]) - 0.5) < 0.001
    assert float(dropped.position[2]) < 0.3
    assert np.min(telemetry.normal[-1]) > 30
    assert np.max(low.slip_speed) > 0.02
    assert not np.any(low.friction)
    np.testing.assert_allclose(held.rotation.T @ held.rotation, np.eye(3), atol=1e-9)


def test_weak_grip_yields_instead_of_welding_object():
    closed, _ = simulate(initial(), 0.8, 40.0, jnp.array([[0.2, 0.0, 0.2, 0.2, 0.47]]))
    dropped, report = simulate(closed, 0.8, 0.25, jnp.empty((0, 5)))
    assert float(dropped.position[2]) < 0.3
    assert np.max(report.slip_speed) > 0.01


def test_coulomb_cone_and_no_tension_or_contact_memory_in_air():
    normal = jnp.array([[0.0, 1.0, 0.0]])
    velocity = jnp.array([[1.0, 0.0, 0.0]])
    force, memory, fn, ft, speed, sliding = friction_force(
        normal, jnp.array([0.001]), velocity, jnp.array([[0.1, 0.0, 0.0]]), 0.0002, 0.8, 0.6
    )
    assert bool(sliding[0]) and float(ft[0]) == pytest.approx(0.6 * float(fn[0]))
    assert float(jnp.sum(force * velocity)) < 0
    force, memory, *_ = friction_force(
        normal, jnp.array([-0.001]), velocity, memory, 0.0002, 0.8, 0.6
    )
    np.testing.assert_array_equal(force, 0)
    np.testing.assert_array_equal(memory, 0)


def test_contact_timestep_refinement_preserves_supported_pose():
    supports = jnp.array([[0.2, 0.0, 0.2, 0.2, 0.47]])
    coarse, _ = simulate(initial(), 0.8, 40.0, supports)
    fine, _ = simulate(initial(), 0.8, 40.0, supports, dt=0.0001)
    fine, _ = simulate(fine, 0.8, 40.0, supports, dt=0.0001)
    np.testing.assert_allclose(coarse.position, fine.position, atol=1e-4)
    np.testing.assert_allclose(coarse.width, fine.width, atol=1e-4)


def test_support_edge_closest_point_matches_analytic_distance():
    from robogogogo.simulation.grasp_contact import closest_segment_box_parameter

    a, b = jnp.array([0.0, -0.2, 0.4]), jnp.array([0.0, 0.2, 0.6])
    lower, upper = jnp.array([[-0.1, -0.05, 0.0]]), jnp.array([[0.1, 0.05, 0.45]])
    t = closest_segment_box_parameter(a, b, lower, upper)[0]
    point = a + t * (b - a)
    distance = jnp.linalg.norm(point - jnp.clip(point, lower[0], upper[0]))
    from robogogogo.planning.collision import segment_box_distance

    expected = segment_box_distance(
        np.array([a]), np.array([b]), np.array([0.0, 0.0, 0.225]), np.array([0.2, 0.1, 0.45])
    )[0]
    assert float(distance) == pytest.approx(expected, abs=1e-10)


def test_articulated_arm_receives_contact_reaction_without_loaded_model():
    from robogogogo.manipulation import Manipulation
    from robogogogo.session import Session

    builder = sp.ModelBuilder()
    parent = -1
    for i, axis in enumerate(np.eye(3)):
        parent = builder.add_link(
            f"joint_{i}",
            parent=parent,
            joint="prismatic",
            axis=axis,
            origin=(0, 0, 0.5) if i == 0 else (0, 0, 0),
            mass=1.0,
            inertia=np.eye(3) * 0.01,
            limits=(-1, 1),
            effort=100.0,
        )
    model = builder.build()
    session = Session(model, arms=[dict(name="arm", tool="joint_2")])
    m = Manipulation(model, session.arms, [0.2, 0, 0.5], [])
    m.contact = m.contact._replace(
        rotation=initial().rotation, width=jnp.array([0.0555]), drive_width=jnp.array([0.0555])
    )
    session.contact_system = m
    m.grasp("arm", session)
    for _ in range(70):
        session.advance()
    assert m.owner == "arm"
    assert session.payload_model is None
    assert float(m.last_reaction[2]) < -2.0
    assert float(m.state.position[2]) > 0.49
    assert m.grippers["arm"]["holding"]


def test_receiver_without_contact_never_opens_the_giver():
    from robogogogo.manipulation import Manipulation
    from robogogogo.session import Session

    builder = sp.ModelBuilder()
    for name, x in [("giver", 0.0), ("receiver", 2.0)]:
        builder.add_link(
            name,
            joint="prismatic",
            axis=(0, 0, 1),
            origin=(x, 0, 0.5),
            mass=1.0,
            inertia=np.eye(3) * 0.01,
            limits=(-1, 1),
            effort=100.0,
        )
    session = Session(builder.build(), arms=[dict(name=n, tool=n) for n in ["giver", "receiver"]])
    m = Manipulation(session.model, session.arms, [0.2, 0, 0.5], [])
    m.contact = m.contact._replace(
        rotation=initial().rotation,
        width=jnp.array([0.0555, 0.1]),
        drive_width=jnp.array([0.0555, 0.1]),
    )
    session.contact_system = m
    m.grasp("giver", session)
    for _ in range(70):
        session.advance()
    assert m.owner == "giver"
    m.transfer("receiver", session)
    for _ in range(330):
        session.advance()
    assert m.failure and "no sustained stable contact" in m.failure
    assert m.owner == "giver" and m.grippers["giver"]["command"] == "closed"
    assert float(m.state.position[2]) > 0.49
    assert max(m.grippers["receiver"]["normal_force_n"]) == 0
    assert not any(e["event"] == "handover" for e in m.events)


def test_loaded_support_resists_rolling_without_air_drag():
    state = initial()._replace(
        position=jnp.array([0.2, 0.0, 0.03]),
        rotation=jnp.eye(3),
        velocity=jnp.array([0.03, 0.0, 0.0]),
        angular_velocity=jnp.array([0.0, 1.0, 0.0]),
    )
    result, _ = simulate(state, 0.8, 40.0, jnp.empty((0, 5)))
    obj = GraspObject()
    kinetic = 0.5 * obj.mass * 0.03**2 + 0.5 * obj.inertia[1, 1]
    expected_distance = kinetic / (0.01 * obj.mass * 9.81)
    assert float(result.position[0]) - 0.2 == pytest.approx(expected_distance, abs=3e-4)
    assert np.linalg.norm(result.angular_velocity) < 0.05
    airborne = state._replace(position=jnp.array([0.2, 0.0, 5.0]))
    result, _ = simulate(airborne, 0.8, 40.0, jnp.empty((0, 5)))
    np.testing.assert_allclose(result.angular_velocity, [0.0, 1.0, 0.0], atol=1e-6)


def test_fast_unload_preserves_measured_overload_and_actuator_limit():
    state = initial()._replace(
        width=jnp.array([0.058]), drive_width=jnp.array([0.050]), support_tangent=jnp.zeros((3, 3))
    )
    obj = GraspObject()
    for limit in (0.0, 10.0):
        args = (
            state,
            jnp.array([[0.0, 0.0, 0.5]]),
            jnp.eye(3)[None],
            jnp.zeros((1, 3)),
            jnp.zeros((1, 3)),
            jnp.array([0.015]),
            jnp.array([limit]),
            jnp.array([0.8]),
            jnp.array([0.6]),
            obj.mass,
            jnp.asarray(obj.inertia),
            obj.radius,
            obj.half_length,
            jnp.array([0.0, 0.0, -9.81]),
            jnp.empty((0, 5)),
            0.0002,
        )
        regular, report = contact_step(*args)
        unloaded, protected = contact_step(*args, force_ceiling=jnp.array([2.0]))
        assert float(protected.normal.max()) > 2.0  # Never hide the impact by clipping it.
        np.testing.assert_allclose(report.normal, protected.normal)
        assert float(unloaded.drive_width[0]) > float(regular.drive_width[0])
        if limit == 0:
            np.testing.assert_allclose(unloaded.width_velocity, regular.width_velocity)
        else:
            assert float(unloaded.width_velocity[0]) > float(regular.width_velocity[0])
