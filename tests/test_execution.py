"""Command lifecycle tests use a fake clock and no physical hardware."""

import threading
import time
from dataclasses import replace

import pytest

from robogogogo.robot.execution import CoordinatedGoal, GripperGoal
from robogogogo.robot.test_execution import TestExecutionBackend
from robogogogo.runtime.execution import CommandExecutor


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, amount):
        self.now += amount


def setup():
    clock = Clock()
    backend = TestExecutionBackend(clock=clock)
    executor = CommandExecutor(backend, clock=clock)
    return clock, backend, executor


def goal():
    return CoordinatedGoal((0.2,) * 6, (-0.2,) * 6, GripperGoal(0.02, 5), GripperGoal(0.04, 8), 2.0)


def submit(executor, lease, target=None, sequence=1, observation=None):
    return executor.submit(
        lease["token"],
        sequence,
        observation if observation is not None else executor.snapshot()["feedback"]["sequence"],
        target or goal(),
    )


def advance(clock, executor, lease, duration):
    for _ in range(round(duration / 0.1)):
        clock.advance(0.1)
        executor.heartbeat(lease["token"])


def test_atomic_dual_arm_gripper_progress_completion_and_force_caps():
    clock, backend, executor = setup()
    assert executor.snapshot()["stop_confirmed"]
    lease = executor.arm("policy")
    result = submit(executor, lease)
    assert backend.accepted == [result["command_id"]]
    assert backend.force_caps == (5, 8)
    advance(clock, executor, lease, 1.0)
    sample = executor.snapshot()["feedback"]
    assert sample["left"] == pytest.approx([0.1] * 6)
    assert sample["right"] == pytest.approx([-0.1] * 6)
    assert sample["grippers"] == pytest.approx([0.04, 0.05])
    advance(clock, executor, lease, 1.1)
    status = executor.snapshot()
    assert status["mode"] == "armed" and status["active_command"] is None
    assert status["feedback"]["left"] == pytest.approx([0.2] * 6)
    assert status["feedback"]["grippers"] == pytest.approx([0.02, 0.04])
    assert status["events"][-1]["kind"] == "completed"
    assert "token" not in str(status)


@pytest.mark.parametrize(
    "bad",
    [
        replace(goal(), left=(0.0,)),
        replace(goal(), left=(float("nan"),) * 6),
        replace(goal(), right=(2.0,) * 6),
        replace(goal(), duration_s=0.1),
        replace(goal(), left_gripper=GripperGoal(0.1, 5)),
        replace(goal(), right_gripper=GripperGoal(0.04, 100)),
    ],
)
def test_invalid_group_never_partially_executes(bad):
    _, backend, executor = setup()
    lease = executor.arm("manual")
    with pytest.raises(ValueError):
        submit(executor, lease, bad)
    assert backend.accepted == []
    assert backend.position == [0.0] * 12 + [0.06, 0.06]


def test_ownership_busy_replay_and_stale_observations():
    clock, _, executor = setup()
    lease = executor.arm("policy")
    with pytest.raises(ValueError):
        executor.arm("manual")
    with pytest.raises(ValueError, match="lease"):
        submit(executor, {"token": "wrong"})
    old = executor.snapshot()["feedback"]["sequence"]
    clock.advance(0.3)
    with pytest.raises(ValueError, match="stale observation"):
        submit(executor, lease, observation=old)
    submit(executor, lease)
    with pytest.raises(ValueError, match="already executing"):
        submit(executor, lease, sequence=2)
    advance(clock, executor, lease, 2.1)
    with pytest.raises(ValueError, match="sequence"):
        submit(executor, lease, sequence=1)


def test_takeover_stops_both_arms_preserves_grip_and_revokes_policy():
    clock, _, executor = setup()
    policy = executor.arm("policy")
    submit(executor, policy)
    advance(clock, executor, policy, 0.5)
    before = executor.snapshot()["feedback"]
    manual = executor.takeover()
    status = manual["snapshot"]
    assert status["owner"] == "manual" and status["mode"] == "armed"
    assert not status["feedback"]["moving"]
    assert status["feedback"]["left"] == before["left"]
    assert status["feedback"]["grippers"] == before["grippers"]
    for action in (lambda: submit(executor, policy), lambda: executor.heartbeat(policy["token"])):
        with pytest.raises(ValueError, match="lease"):
            action()
    submit(executor, manual)
    executor.stop()
    with pytest.raises(ValueError, match="lease"):
        submit(executor, manual, sequence=2)


def test_owner_watchdog_requires_explicit_fault_reset_and_cannot_be_revived():
    clock, _, executor = setup()
    lease = executor.arm("policy")
    submit(executor, lease)
    clock.advance(1.01)
    with pytest.raises(ValueError, match="lease"):
        executor.heartbeat(lease["token"])
    status = executor.snapshot()
    assert status["mode"] == "fault" and status["stop_confirmed"]
    assert "watchdog" in status["reason"]
    with pytest.raises(ValueError):
        executor.arm("manual")
    executor.reset()
    assert executor.arm("manual")["epoch"] > lease["epoch"]


def test_backend_lease_clamps_motion_without_supervisor_tick():
    clock, backend, executor = setup()
    lease = executor.arm("policy")
    submit(executor, lease)
    clock.advance(4.0)
    feedback = backend.read()
    assert feedback.fault == "Backend lease expired"
    assert not feedback.moving
    assert feedback.left == pytest.approx([0.1] * 6)
    assert feedback.grippers == pytest.approx([0.04, 0.05])


def test_stale_feedback_fault_and_unconfirmed_stop_cannot_rearm():
    clock, backend, executor = setup()
    lease = executor.arm("policy")
    submit(executor, lease)
    backend.freeze_feedback = True
    clock.advance(0.3)
    executor.tick()
    status = executor.snapshot()
    assert status["mode"] == "fault" and not status["stop_confirmed"]
    with pytest.raises(ValueError):
        executor.reset()
    backend.freeze_feedback = False
    executor.reset()
    executor.arm("manual")
    backend.fail_stop = True
    status = executor.stop()
    assert status["mode"] == "fault" and not status["stop_confirmed"]
    with pytest.raises(ValueError):
        executor.takeover()


def test_rejection_and_stalled_command_stop_whole_group():
    clock, backend, executor = setup()
    lease = executor.arm("policy")
    backend.reject_next = True
    with pytest.raises(ValueError, match="acceptance"):
        submit(executor, lease)
    assert backend.accepted == [] and executor.snapshot()["mode"] == "fault"
    executor.reset()
    lease = executor.arm("policy")
    backend.stall_motion = True
    submit(executor, lease)
    for _ in range(25):
        clock.advance(0.1)
        if executor.snapshot()["mode"] == "fault":
            break
        executor.heartbeat(lease["token"])
    status = executor.snapshot()
    assert status["mode"] == "fault" and "completion watchdog" in status["reason"]
    assert status["stop_confirmed"]


def test_concurrent_takeover_and_old_owner_submission_linearize():
    _, backend, executor = setup()
    lease = executor.arm("policy")
    barrier = threading.Barrier(2)

    def old_owner():
        barrier.wait()
        try:
            submit(executor, lease)
        except ValueError:
            pass

    thread = threading.Thread(target=old_owner)
    thread.start()
    barrier.wait()
    manual = executor.takeover()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert executor.snapshot()["owner"] == "manual"
    assert backend.trajectory is None
    submit(executor, manual)


def test_background_watchdog_runs_without_status_polling_and_shutdown_stops():
    backend = TestExecutionBackend()
    executor = CommandExecutor(backend)
    executor.lease_s = 0.08
    executor.start()
    executor.arm("policy")
    time.sleep(0.14)
    assert executor.mode == "fault"  # Inspect state without invoking a polling method.
    executor.close()
    assert backend.closed and executor.mode == "closed" and executor.token is None


def test_invalid_gripper_feedback_and_uncommissioned_adapters_are_rejected():
    _, backend, executor = setup()
    sample = backend.read()
    for velocity in ((float("nan"), 0.0), (0.5, 0.0)):
        with pytest.raises(ValueError):
            executor._validate_feedback(replace(sample, gripper_velocity=velocity))
    backend.is_test = False
    with pytest.raises(ValueError, match="commissioning"):
        CommandExecutor(backend)


def test_delayed_arming_and_takeover_cannot_override_a_newer_stop():
    _, _, executor = setup()
    old_epoch = executor.snapshot()["epoch"]
    executor.stop()
    with pytest.raises(ValueError, match="changed"):
        executor.arm("policy", expected_epoch=old_epoch)
    with pytest.raises(ValueError, match="changed"):
        executor.takeover(expected_epoch=old_epoch)
    assert executor.snapshot()["mode"] == "stopped"


def test_stop_does_not_confirm_a_moving_gripper():
    _, backend, executor = setup()
    stopped = backend.stop
    backend.stop = lambda: replace(stopped(), gripper_velocity=(0.01, 0.0))
    state = executor.stop()
    assert state["mode"] == "fault" and not state["stop_confirmed"]


def test_wrong_command_acknowledgement_faults_the_entire_group():
    _, backend, executor = setup()
    lease = executor.arm("policy")
    submit(executor, lease)
    backend.command_id = "obsolete-command"
    executor.tick()
    state = executor.snapshot()
    assert state["mode"] == "fault" and state["stop_confirmed"]
    assert "acknowledgement" in state["reason"]
