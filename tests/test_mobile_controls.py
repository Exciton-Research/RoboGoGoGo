"""Exercise the same joint, wheel and recorded rule-based paths exposed by Studio."""

import numpy as np
import pytest
from test_vla import png_observation

from robogogogo.studio.schemas import CameraObservation, Command, NewEpisode, PolicyAction
from robogogogo.studio.workspace import Studio


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    return Studio(tmp_path_factory.mktemp("mobile-controls"))


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_recorded_rule_based_reach_finishes(studio, variant):
    studio.episode_service.start_episode(NewEpisode(variant=variant, camera_preset="head_wrists"))
    initial = np.asarray(studio.session.state.q).copy()
    images = {camera["name"]: png_observation() for camera in studio.episode.cameras}

    def capture():
        return studio.episode_service.capture_observation(
            CameraObservation(sequence=studio.sequence, images=images)
        )["observation"]

    index = capture()
    for _ in range(80):
        proposal = studio.episode_service.rule_action()
        studio.episode_service.policy_action(
            PolicyAction(
                delta=proposal["delta"], source="rule_based", observation=index, control_ticks=10
            )
        )
        index = capture()
        if studio.episode.done:
            break
    assert studio.error is None
    assert studio.episode.success
    assert len(studio.episode.observations) == len(studio.episode.actions) + 1
    assert np.max(np.abs(np.asarray(studio.session.state.q) - initial)) > 0.05


def test_joint_button_runs_and_drive_ownership_releases(studio):
    studio.command(Command(op="reset"))
    arm = studio.session.arms[0]
    goal = np.asarray(studio.session.state.q)[list(arm.joint_indices)].copy()
    goal[3] += 0.12
    studio.command(Command(op="joints", arm=arm.name, values=goal.tolist(), start=True))
    assert studio.running
    with pytest.raises(ValueError, match="Finish or cancel"):
        studio.command(Command(op="drive", values=[0.2, 0, 0], start=True))
    for _ in range(250):
        studio.tick()
        if studio.session.motion_plan["status"] == "complete":
            break
    assert studio.session.motion_plan["status"] == "complete"
    assert abs(float(studio.session.state.q[arm.joint_indices[3]]) - goal[3]) < 0.01
    studio.command(Command(op="pause"))
    assert not studio.running


@pytest.mark.parametrize(
    "twist,axis,sign",
    [
        ([0.2, 0, 0], 0, 1),
        ([-0.2, 0, 0], 0, -1),
        ([0, 0.15, 0], 1, 1),
        ([0, -0.15, 0], 1, -1),
        ([0, 0, 0.2], 2, 1),
        ([0, 0, -0.2], 2, -1),
    ],
)
def test_drive_buttons_move_base_and_brake(studio, twist, axis, sign):
    studio.command(Command(op="reset"))
    studio.command(Command(op="drive", values=twist, duration=3, start=True))
    assert studio.running
    for _ in range(150):
        studio.tick()
    assert studio.error is None
    assert float(studio.session.state.q[axis]) * sign > 0.03
    assert np.max(np.abs(np.asarray(studio.session.state.q)[studio.session.mobile.wheels])) > 0.1
    studio.command(Command(op="brake"))
    for _ in range(75):
        studio.tick()
    assert np.max(np.abs(np.asarray(studio.session.state.qd)[studio.session.mobile.base])) < 0.035
    studio.command(Command(op="pause"))


def test_failed_simulation_cannot_be_started_by_motion_button(studio):
    studio.command(Command(op="reset"))
    studio.error = "test fault"
    with pytest.raises(ValueError, match="Reset"):
        studio.command(Command(op="drive", values=[0.2, 0, 0], start=True))
    assert not studio.running
    np.testing.assert_array_equal(studio.session.drive_twist, 0)
    studio.command(Command(op="reset"))
