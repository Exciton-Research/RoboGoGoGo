"""Structural objects interact with robot dynamics, load-transfer and damage guards."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from robogogogo.studio.schemas import PlanTask

BENCH = runpy.run_path(str(Path(__file__).parents[1] / "examples/structural_handover.py"))


def test_structural_damage_feedback_unloads_before_damage():
    protected = BENCH["run"]("soft_foam", 40.0, lift=0.0, duration=3.0, transfer=False)
    unsafe = BENCH["run"]("soft_foam", 40.0, lift=0.0, duration=1.0, transfer=False, adaptive=False)
    assert protected["failure"] is None
    assert not protected["object"]["material"]["damaged"]
    assert protected["maximum_risk"] > 0.55
    assert protected["adaptive_grasp"]["left"]["force_setpoint_n"] < 25
    assert any(e["event"] == "adaptive_unloading" for e in protected["events"])
    assert unsafe["object"]["material"]["damaged"]
    assert unsafe["failure"] is not None


def test_structural_supported_handover_verifies_load_before_release():
    result = BENCH["run"]("soft_foam", 4.0, lift=0.0, duration=7.0)
    assert result["failure"] is None
    assert result["object"]["owner"] == "right"
    assert not result["object"]["material"]["damaged"]
    events = [e["event"] for e in result["events"]]
    assert (
        events.index("load_transfer_started")
        < events.index("receiver_load_verified")
        < events.index("open_command")
        < events.index("handover")
    )
    assert len(result["object"]["structure"]["fragments"]) == 1
    assert np.linalg.norm(result["reaction_torque"]) > 0


def test_structural_envelope_contains_all_nodes_and_pose_setter_moves_nodes():
    session, grasp = BENCH["make_rig"]()
    state = grasp.state
    grasp.state = state._replace(position=state.position + np.array([0.02, 0.03, 0.04]))
    points = np.asarray(grasp.contact.structure.position)
    np.testing.assert_allclose(points.mean(0), grasp.state.position)
    bound = grasp.object_segment()
    assert bound.shape == (5, 3)
    axis = bound[1] - bound[0]
    t = np.clip((points - bound[0]) @ axis / np.dot(axis, axis), 0, 1)
    assert np.max(np.linalg.norm(points - bound[0] - t[:, None] * axis, axis=1)) <= bound[2, 0]
    np.testing.assert_allclose(bound[3], points.min(0))
    np.testing.assert_allclose(bound[4], points.max(0))
    assert session.payload_model is None


def test_structural_task_requires_material():
    with pytest.raises(ValueError, match="deformable material"):
        PlanTask(kind="handover", structural_dynamics=True)
    assert PlanTask(
        kind="handover", structural_dynamics=True, object_material="soft_foam"
    ).structural_dynamics


def test_lifted_elastomer_handover_retains_connected_object():
    result = BENCH["run"]("soft_elastomer", 16.0, lift=0.1, duration=12.0)
    assert result["failure"] is None
    assert result["object"]["owner"] == "right"
    assert result["object"]["position"][2] > 0.57
    assert result["object"]["structure"]["broken_bonds"] == 0
    assert result["maximum_bend_m"] > 0.0001
    assert any(e["event"] == "receiver_load_verified" for e in result["events"])
    assert any(e["event"] == "adaptive_unloading" for e in result["events"])


def test_structural_collision_uses_live_radius_without_mutating_nominal_model():
    from robogogogo.planning import Capsule, CollisionWorld, SphereObstacle

    session, grasp = BENCH["make_rig"]()
    world = CollisionWorld(
        session.model,
        [Capsule("payload", 2, (0, 0, 0), (0, 0, 0), 0.03)],
        obstacles=[SphereObstacle("probe", (0.29, 0, 0.5), 0.01)],
    )
    bound = grasp.object_segment()
    ordinary = world.clearance(session.state.q, bound)
    enlarged = bound.copy()
    enlarged[2, 0] += 0.08
    collision = world.clearance(session.state.q, enlarged)
    assert ordinary.distance > 0
    assert collision.distance < 0
    assert collision.pair == ("payload", "probe")
    assert world.radii[0] == 0.03


def test_damage_blocks_pending_receiver_release():
    import jax.numpy as jnp

    session, grasp = BENCH["make_rig"]()
    grasp.owner = "left"
    grasp.pending = ("transfer", "right", "left", session.elapsed)
    grasp.grippers["right"]["command"] = "closed"
    grasp.contact = grasp.contact._replace(
        structure=grasp.contact.structure._replace(
            damage=jnp.ones_like(grasp.contact.structure.damage)
        )
    )
    session.advance()
    assert grasp.failure == "Object fractured"
    assert grasp.owner == "left"
    assert grasp.grippers["left"]["command"] == "closed"
    assert not any(e["event"] in ("open_command", "handover") for e in grasp.events)
