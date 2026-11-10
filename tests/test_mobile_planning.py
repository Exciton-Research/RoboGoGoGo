"""Real robot detours, planning failure atomicity and segment execution."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from test_mobile_self_collision import EDGE_A, EDGE_B, TORSO, configuration

from robogogogo.planning.mobile import plan_mobile
from robogogogo.planning.motion import PlanningError
from robogogogo.robot.mobile_design import mobile_arm_definitions, mobile_dual_arm_urdf, mobile_home
from robogogogo.session import Session
from robogogogo.simulation.mobile import MobileDynamics
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    m = loads(mobile_dual_arm_urdf()).model
    return replace(m, home=jnp.asarray(mobile_home(m)))


def test_real_blocked_edge_gets_certified_detour(model):
    dynamics = MobileDynamics(model, 5)
    a, b = configuration(model, EDGE_A), configuration(model, EDGE_B)
    assert not dynamics.collision_world.edge_free(a, b)
    plan = plan_mobile(dynamics, a, b, seed=0)
    assert plan.method in ("rrt_connect", "neutral_detour")
    assert len(plan.waypoints) > 2
    np.testing.assert_allclose(plan.waypoints[0], a)
    np.testing.assert_allclose(plan.waypoints[-1], b)
    for start, end in zip(plan.waypoints, plan.waypoints[1:]):
        assert dynamics.collision_world.edge_free(start, end)
        dynamics.validate_goal(np.asarray(start), np.asarray(end))
        np.testing.assert_array_equal(np.asarray(end)[dynamics.driven], b[dynamics.driven])
    with pytest.raises(PlanningError, match="budget"):
        plan_mobile(dynamics, a, b, budget_seconds=0)


def test_auto_plan_executes_then_releases_drive_and_cancel_is_explicit(model):
    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    goal = np.asarray(model.home).copy()
    goal[model.indices[model.names.index("left_upper_roll")]] += 0.08
    session.set_joints(goal)
    assert session.motion_plan["status"] == "ready"
    for _ in range(400):
        session.advance()
        if session.motion_plan["status"] == "complete":
            break
    assert session.motion_plan["status"] == "complete"
    assert session.mobile_status["collision_guard"] == "clear"
    assert abs(float(session.state.q[16]) - 0.08) < 0.01
    session.set_drive([0.1, 0, 0])
    session.stop_drive()
    before = session.target_q
    with pytest.raises(PlanningError, match="Self-collision"):
        invalid = configuration(model, TORSO)
        for name in ("head_pan", "head_tilt"):
            index = model.indices[model.names.index(name)]
            invalid[index] = float(session.state.q[index])
        invalid[session.mobile.driven] = np.asarray(session.state.q)[session.mobile.driven]
        session.set_joints(invalid)
    np.testing.assert_array_equal(session.target_q, before)
    goal = np.asarray(session.state.q).copy()
    goal[16] = 0.1
    session.set_joints(goal)
    session.cancel_motion_plan()
    assert session.motion_plan["status"] == "cancelled"
    assert not session.motion_waypoints
