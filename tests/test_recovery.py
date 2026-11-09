"""Physical recovery regressions plus guardrails for unsafe attempts."""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo.planning import Capsule, CollisionWorld, SphereObstacle
from robogogogo.planning.collision import CollisionViolation
from robogogogo.studio.schemas import PlanTask

# Load the runnable benchmark and its adjacent fixture without a package install.
DIRECTORY = Path(__file__).parents[1] / "examples"
for name in ("structural_handover", "grasp_recovery"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
BENCH = sys.modules["grasp_recovery"]


def test_regrasp_keeps_helper_closed_and_resumes_without_damage():
    result = BENCH.run("regrasp")
    assert result["failure"] is None
    assert result["recovery"]["phase"] == "regrasped"
    assert result["minimum_helper_load_n"] > 0.3
    assert not result["object"]["material"]["damaged"]
    assert result["grippers"]["left"]["holding"] and result["grippers"]["right"]["holding"]
    assert not result["motion_hold"]
    events = result["events"]
    assert any(e["event"] == "recovery_opening" for e in events)
    assert any(e["event"] == "recovery_reclosing" for e in events)
    assert not any(e["event"] == "open_command" and e["arm"] == "right" for e in events)


def test_damage_put_down_preserves_damage_and_verifies_physical_support():
    result = BENCH.run("put_down")
    assert result["failure"] is None
    assert result["recovery"]["phase"] == "placed"
    assert result["object"]["owner"] is None
    assert result["object"]["material"]["damage_fraction"] >= 0.02
    assert not result["object"]["material"]["fractured"]
    assert result["support_load_n"][1] > 0.8 * 0.16 * 9.81
    assert np.linalg.norm(result["object"]["velocity"]) < 0.01
    assert all(g["command"] == "open" and not g["holding"] for g in result["grippers"].values())
    events = {e["event"]: e["time"] for e in result["events"]}
    assert (
        events["recovery_lowering"]
        < events["recovery_seating"]
        < events["recovery_releasing"]
        < events["recovery_placed"]
    )
    assert events["recovery_seating"] - events["recovery_lowering"] > 1.0


def test_warning_put_down_avoids_damage():
    result = BENCH.run("warning")
    assert result["failure"] is None
    assert result["recovery"]["phase"] == "placed"
    assert result["recovery"]["reason"] == "material_risk"
    assert not result["object"]["material"]["damaged"]


def test_missing_support_cannot_authorize_release():
    session, grasp = BENCH.make_rig()
    grasp.enable_recovery()
    grasp.state = grasp.state._replace(position=jnp.array([2.0, 2.0, 0.7]))
    grasp.recovery.begin_lowering(grasp, session)
    assert grasp.recovery.phase == "blocked"
    assert "No registered support" in grasp.failure
    assert grasp.grippers["left"]["command"] == "closed"


def test_recovery_does_not_bypass_collision_guard():
    session, grasp = BENCH.make_rig()
    grasp.enable_recovery()
    grasp.grippers["left"]["holding"] = True  # Set up the policy branch; collisions still use FK.
    grasp.recovery.begin_lowering(grasp, session)
    tool = np.asarray(session.poses().position)[2]
    world = CollisionWorld(
        session.model,
        [Capsule("tool", 2, (0, 0, 0), (0, 0, 0), 0.02)],
        obstacles=[SphereObstacle("blocked_route", tuple(tool), 0.02)],
    )
    before = np.asarray(session.state.q).copy()
    with pytest.raises(CollisionViolation):
        session.advance(collision_world=world)
    np.testing.assert_array_equal(session.state.q, before)
    assert session.elapsed == 0
    assert grasp.recovery.phase == "blocked"
    assert not any(e["event"] == "open_command" for e in grasp.events)


def test_open_command_removes_stale_closing_drive_without_teleporting_jaws():
    session, grasp = BENCH.make_rig()
    grasp.contact = grasp.contact._replace(drive_width=jnp.array([0.015, 0.1]))
    before = np.asarray(grasp.contact.width).copy()
    grasp._command("left", "open", session)
    np.testing.assert_array_equal(grasp.contact.width, before)
    assert grasp.contact.drive_width[0] == before[0]
    assert grasp.contact.structure.drive_width[0] == before[0]


def test_recovery_requires_force_feedback():
    with pytest.raises(ValueError, match="adaptive force feedback"):
        PlanTask(recovery_behaviors=True, adaptive_grasp=False)
