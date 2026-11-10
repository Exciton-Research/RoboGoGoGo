"""Grasp preconditions, continuous ownership, payload inertia and support contact."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.embodiment import ArmGroup
from robogogogo.manipulation import (
    GraspObject,
    Manipulation,
    ObjectState,
    free_object_step,
    payload_model,
)
from robogogogo.planning.collision import segment_box_distance


def rig():
    builder = sp.ModelBuilder()
    for name in ("giver", "receiver"):
        builder.add_link(
            name,
            joint="prismatic",
            axis=(1, 0, 0),
            mass=1,
            inertia=np.eye(3) * 0.01,
            limits=(-1, 1),
        )
    model = builder.build()
    arms = tuple(
        ArmGroup(name, name, index, (index,), (name,)) for index, name in enumerate(model.names)
    )
    position = np.array([0.75, 0, 0.65])
    rotation = np.asarray(sp.rpy_matrix(jnp.array([np.pi / 2, 0, 0])))
    positions = np.array(
        [position - rotation @ np.array([0.20, 0, sign * 0.09]) for sign in (1, -1)]
    )
    pose = sp.Pose(jnp.asarray(positions), jnp.asarray([rotation, rotation]))
    session = SimpleNamespace(
        state=sp.State(jnp.zeros(2), jnp.zeros(2)),
        elapsed=0.2,
        payload_model=None,
        poses=lambda: pose,
    )
    manipulation = Manipulation(model, arms, position, [[0.75, 0, 0.1, 0.1, 0.62]])
    return model, manipulation, session, pose


def test_grasp_command_does_not_attach_or_teleport_object():
    _, m, s, _ = rig()
    before = np.asarray(m.state.position).copy()
    m.grasp("giver", s)
    assert m.owner is None and not m.grippers["giver"]["holding"]
    assert m.grippers["giver"]["command"] == "closed"
    assert s.payload_model is None
    np.testing.assert_array_equal(m.state.position, before)
    with pytest.raises(ValueError, match="held object"):
        m.transfer("receiver", s)
    assert m.grippers["receiver"]["command"] == "open"


def test_contact_parameters_and_unknown_commands_are_validated_atomically():
    _, m, s, _ = rig()
    with pytest.raises(ValueError, match="Unknown"):
        m.grasp("missing", s)
    assert m.pending is None
    with pytest.raises(ValueError, match="dynamic"):
        m.configure_gripper("giver", mu_static=0.1)
    assert m.grippers["giver"]["mu_static"] == 0.8
    # Weak or frictionless grasps are simulated, not rejected by a static force gate.
    m.configure_gripper("giver", force_limit_n=0, mu_static=0, mu_dynamic=0)
    m.grasp("giver", s)
    assert m.owner is None and s.payload_model is None


def test_payload_inertia_changes_native_mass_matrix_and_gravity():
    from fixtures.models import arm6

    model = arm6()
    obj = GraspObject()
    loaded = payload_model(
        model, len(model.names) - 1, obj.mass, jnp.array([0.2, 0, 0.09]), obj.inertia
    )
    delta = np.asarray(sp.mass_matrix(loaded, model.home) - sp.mass_matrix(model, model.home))
    assert np.linalg.eigvalsh(delta).min() > -1e-9
    assert np.linalg.norm(delta) > 0.01
    base_g = sp.inverse_dynamics(model, model.home, jnp.zeros(6), jnp.zeros(6))
    load_g = sp.inverse_dynamics(loaded, model.home, jnp.zeros(6), jnp.zeros(6))
    assert np.linalg.norm(load_g - base_g) > 0.1


def test_free_object_falls_and_settles_on_support_without_teleporting():
    state = ObjectState(jnp.array([0.0, 0.0, 1.0]), jnp.eye(3), jnp.zeros(3), jnp.zeros(3))
    supports = jnp.array([[0.0, 0.0, 0.1, 0.1, 0.4]])
    next_state = free_object_step(state, 0.01, jnp.array([0.0, 0.0, -9.81]), 0.03, 0.12, supports)
    assert 0.99 < float(next_state.position[2]) < 1
    for _ in range(200):
        state = free_object_step(state, 0.01, jnp.array([0.0, 0.0, -9.81]), 0.03, 0.12, supports)
    assert float(state.position[2]) == pytest.approx(0.43)
    assert float(state.velocity[2]) == 0
    np.testing.assert_allclose(np.asarray(state.rotation).T @ state.rotation, np.eye(3), atol=1e-9)


def test_segment_box_distance_crossing_parallel_and_point():
    start = np.array([[-2, 0, 0], [-2, 2, 0], [2, 2, 0], [2, 0, 0]], dtype=float)
    end = np.array([[2, 0, 0], [2, 2, 0], [2, 2, 0], [3, 0, 0]], dtype=float)
    np.testing.assert_allclose(
        segment_box_distance(start, end, [0, 0, 0], [2, 2, 2]), [0, 1, np.sqrt(2), 1]
    )


def test_free_object_without_supports_lands_on_floor():
    state = ObjectState(jnp.array([0.0, 0.0, 0.031]), jnp.eye(3), jnp.zeros(3), jnp.zeros(3))
    result = free_object_step(
        state, 0.1, jnp.array([0.0, 0.0, -9.81]), 0.03, 0.12, jnp.empty((0, 5))
    )
    assert float(result.position[2]) == pytest.approx(0.03)
    assert float(result.velocity[2]) == 0
