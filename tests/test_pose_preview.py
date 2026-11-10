"""Draft gestures are read-only; execution remains an explicit guarded command."""

import copy

import numpy as np
import pytest
from test_mobile_self_collision import TORSO, configuration

from robogogogo.studio.schemas import Command
from robogogogo.studio.workspace import Studio


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return Studio(tmp_path_factory.mktemp("pose-preview"))


def test_draft_preview_preserves_robot_plan_and_episode(workspace):
    w = workspace
    q = np.asarray(w.session.state.q).copy()
    draft = q.copy()
    draft[18] += 0.12
    before = copy.deepcopy(w.snapshot())
    result = w.preview_pose(draft)
    assert result["valid"]
    assert result["positions"] != before["state"]["positions"]
    np.testing.assert_array_equal(w.session.state.q, q)
    assert w.snapshot() == before
    assert not w.episode.invalidated
    w.command(Command(op="joints", values=result["values"], start=True))
    for _ in range(250):
        w.tick()
        if w.session.motion_plan["status"] == "complete":
            break
    assert w.error is None and w.session.motion_plan["status"] == "complete"
    assert abs(float(w.session.state.q[18]) - draft[18]) < 0.01
    assert w.episode.invalidated
    w.command(Command(op="reset"))


def test_invalid_preview_is_marked_without_moving_and_frozen_coordinates_stay_fixed(workspace):
    w = workspace
    draft = configuration(w.session.model, TORSO)
    before = np.asarray(w.session.state.q).copy()
    result = w.preview_pose(draft)
    assert not result["valid"]
    np.testing.assert_array_equal(w.session.state.q, before)
    draft = before.copy()
    draft[w.session.mobile.driven] = 99
    result = w.preview_pose(draft)
    np.testing.assert_array_equal(
        np.asarray(result["values"])[w.session.mobile.driven], before[w.session.mobile.driven]
    )
    for values in ([0], [float("nan")] * 31, [100] * 31):
        with pytest.raises(ValueError):
            w.preview_pose(values)


def test_trajectory_export_identifies_assumed_actuator_parameters(workspace, tmp_path):
    path = workspace.session.export(tmp_path / "drive.npz")
    with np.load(path, allow_pickle=False) as recording:
        assert recording["arm_actuator_revision"].item() == "catalog-arm-drive-2"
        assert "not measured" in recording["arm_actuator_provenance"].item()
        speeds = recording["arm_no_load_speed_rad_s"]
        friction = recording["arm_coulomb_friction_nm"]
        assert speeds.shape == friction.shape == (31,)
        assert np.isfinite(speeds).sum() == 14
        assert np.all(friction[np.isfinite(speeds)] > 0)
        assert np.all(friction[~np.isfinite(speeds)] == 0)


def test_harness_rejection_rolls_back_simulation_state(workspace, monkeypatch):
    from robogogogo.planning.collision import CollisionViolation

    workspace.command(Command(op="reset"))
    session = workspace.session
    before = session.state
    elapsed, samples = session.elapsed, len(session.history)
    monkeypatch.setattr(
        session.mobile,
        "harness_report",
        lambda q: dict(
            valid=False, clearance_m=-0.01, pair=["test_service_loop", "arm"], routes=[]
        ),
    )
    with pytest.raises(CollisionViolation, match="Cable routing"):
        session.advance()
    assert session.state is before
    assert session.elapsed == elapsed and len(session.history) == samples
    assert session.mobile_status["collision_guard"] == "blocked"


def test_draft_preview_rejects_wiring_failure_before_apply(workspace, monkeypatch):
    workspace.command(Command(op="reset"))
    session = workspace.session
    before = copy.deepcopy(workspace.snapshot())
    monkeypatch.setattr(
        session.mobile,
        "harness_report",
        lambda q: dict(
            valid=False,
            clearance_m=-0.01,
            pair=["left_yaw_roll_internal", "cover_inner_wall"],
            routes=[],
        ),
    )
    preview = workspace.preview_pose(np.asarray(session.state.q).copy())
    assert not preview["valid"]
    assert preview["pair"] == ["left_yaw_roll_internal", "cover_inner_wall"]
    assert preview["rigid_clearance_m"] >= session.mobile.collision_world.margin
    assert workspace.snapshot() == before
