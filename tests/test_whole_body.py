"""Whole-body reach must move through actuators and retain every runtime guard."""

import numpy as np
import pytest

from robogogogo.planning.motion import PlanningError
from robogogogo.planning.whole_body import WholeBodyPlanner
from robogogogo.studio.schemas import Command
from robogogogo.studio.workspace import Studio


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return Studio(tmp_path_factory.mktemp("whole-body"))


def test_wheels_torso_and_both_arms_reach_together(workspace):
    w, s = workspace, workspace.session
    w.command(Command(op="reset"))
    start = np.asarray(s.state.q).copy()
    planner = WholeBodyPlanner(s.mobile, s.arms)
    target = np.asarray(planner.positions(start)).mean(axis=0) + [0.35, 0.08, -0.12]
    w.command(Command(op="whole_body", values=target.tolist(), start=True))
    assert w.running and s.motion_plan["kind"] == "whole_body"
    np.testing.assert_array_equal(s.state.q, start)  # Planning cannot teleport the base.
    with pytest.raises(ValueError, match="Finish or cancel"):
        s.set_drive([0.2, 0, 0])
    concurrent = False
    for _ in range(900):
        w.tick()
        assert w.error is None
        qd = np.asarray(s.state.qd)
        concurrent |= (
            np.linalg.norm(qd[:2]) > 0.01
            and abs(qd[12]) > 0.01
            and all(np.linalg.norm(qd[list(a.joint_indices)]) > 0.005 for a in s.arms)
        )
        if s.motion_plan["status"] == "complete":
            break
    assert concurrent
    assert s.motion_plan["status"] == "complete"
    assert (
        np.max(
            np.linalg.norm(
                np.asarray(planner.positions(s.state.q)) - s.motion_plan["target_positions"], axis=1
            )
        )
        < 0.02
    )
    assert np.linalg.norm(np.asarray(s.state.q)[:2] - start[:2]) > 0.1
    assert abs(float(s.state.q[12]) - start[12]) > 0.1
    assert s.mobile_status["guard"] == "ok" and s.mobile_status["collision_guard"] == "clear"
    assert np.max(np.abs(np.asarray(s.state.q)[s.mobile.wheels])) > 0.3
    w.command(Command(op="pause"))


def test_rejection_pause_cancel_and_tracking_timeout(workspace):
    w, s = workspace, workspace.session
    w.command(Command(op="reset"))
    center = np.asarray(s.whole_body_planner.positions(s.state.q)).mean(axis=0)
    for goal in ([float("nan"), 0, 1], center + [2, 0, 0], [0, 1]):
        with pytest.raises(PlanningError):
            s.set_whole_body_target(goal)
        assert s.motion_plan is None
    w.command(Command(op="whole_body", values=(center + [0.2, 0, -0.08]).tolist(), start=True))
    for _ in range(30):
        w.tick()
    w.command(Command(op="pause"))
    q, elapsed = np.asarray(s.state.q).copy(), s.elapsed
    w.tick()
    np.testing.assert_array_equal(s.state.q, q)
    assert s.elapsed == elapsed and s.motion_plan["status"] == "paused"
    w.command(Command(op="run"))
    w.tick()
    assert s.elapsed > elapsed
    # Deterministically exercise watchdog expiry without a slow low-traction run.
    s.segment_time = s.elapsed - s.segment_duration - 16
    before = np.asarray(s.state.q).copy()
    w.tick()
    assert "timed out" in w.error and s.motion_plan["status"] == "blocked"
    np.testing.assert_array_equal(s.state.q, before)
    w.command(Command(op="reset"))
    s.set_whole_body_target(center + [0.1, 0, 0])
    w.command(Command(op="cancel_plan"))
    assert s.motion_plan["status"] == "cancelled"
    np.testing.assert_array_equal(s.target_q, s.state.q)
    s.set_drive([0.1, 0, 0])
    w.command(Command(op="reset"))
