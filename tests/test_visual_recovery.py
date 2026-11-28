"""Sensor checks and the single-retry boundary, independent of object pose labels."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robogogogo.demo.controller import PickupFailure
from robogogogo.demo.pretrained import DetectionRejected
from robogogogo.demo.recovery import ClosedLoopController, RecoveryStopped, Retreat
from robogogogo.demo.verification import VisualVerifier


def robot():
    return SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.r_[np.zeros(7), 0.02, 0.02], site_xpos=np.array([[0.4, -0.2, 0.9]])
        ),
        site=0,
        dt=0.05,
        solve=lambda point, joints: joints.copy(),
        target=np.array([0.4, 0.2, 0.8]),
        previous_action=np.r_[np.zeros(7), 0.04],
    )


def detection(x=0.4):
    return {"position": [x, -0.2, 0.8], "grasp_position": [x, -0.2, 0.8], "score": 0.7}


def test_confident_rgb_detection_cannot_confirm_an_object_left_on_the_table():
    pytest.importorskip("cv2")
    sensor = dict(
        rgb=np.zeros((100, 100, 3), np.uint8),
        depth=np.full((100, 100), 1.26),
        position=np.array([0, 0, 2]),
        rotation=np.eye(3),
        fovy=90,
    )
    detector = SimpleNamespace(
        detect=lambda rgb, query, **kwargs: [
            {"box": [0, 0, rgb.shape[1], rgb.shape[0]], "score": 0.9}
        ]
    )
    verifier = VisualVerifier(detector, lambda **kwargs: sensor)
    with pytest.raises(DetectionRejected, match="two camera views"):
        verifier.lift("object", expected=[0, 0, 1.0], opening=0.04)
    sensor["depth"][:] = 1.0  # Same RGB/score, now physically at the lifted height.
    result = verifier.lift("object", expected=[0, 0, 1.0], opening=0.04)
    assert result["views"] == ["side", "front"]
    assert result["position"][2] == pytest.approx(1.0)


def test_gripper_release_and_cell_membership_are_required_for_placement():
    verifier = VisualVerifier(None, None)
    verifier.observe = lambda *args: {
        "position": [0.47, 0.2, 0.8],
        "low": [0.45, 0.18, 0.75],
        "high": [0.49, 0.22, 0.85],
    }
    with pytest.raises(DetectionRejected, match="released"):
        verifier.placement(
            "object", destination=[0.47, 0.2, 0.8], tool=[0.47, 0.2, 1.1], opening=0.02
        )
    with pytest.raises(DetectionRejected, match="outside"):
        verifier.placement(
            "object", destination=[0.35, 0.2, 0.8], tool=[0.47, 0.2, 1.1], opening=0.08
        )


def test_only_one_retry_uses_a_fresh_observation_and_preserves_the_robot_state(monkeypatch):
    sensed = robot()
    before = sensed.data.qpos.copy()
    calls = []

    def find(query, capture, **kwargs):
        calls.append(query)
        return detection(0.43)

    loop = ClosedLoopController(
        sensed,
        query="new object",
        detection=detection(),
        detector=SimpleNamespace(find_target=find),
        capture=lambda **kwargs: None,
    )

    def missed(*args):
        raise PickupFailure("Empty gripper")

    loop.inner.advance = missed
    loop.advance(sensed)
    assert loop.attempt == 2 and loop.stage == "recovering"
    monkeypatch.setattr(Retreat, "advance", lambda self, robot: setattr(self, "done", True))
    loop.advance(sensed)
    assert calls == ["new object"]
    assert loop.pick[0] == pytest.approx(0.43)
    np.testing.assert_array_equal(sensed.data.qpos, before)
    loop.inner.advance = missed
    with pytest.raises(RecoveryStopped, match="No more automatic attempts"):
        loop.advance(sensed)
    assert loop.attempt == 2
    assert len([event for event in loop.events if event["event"] == "retry_started"]) == 1


def test_placement_requires_three_time_separated_stable_observations():
    sensed = robot()
    loop = ClosedLoopController(
        sensed, query="object", detection=detection(), detector=None, capture=None
    )
    loop.inner.phase, loop.inner.elapsed = 8, 0
    times = []
    loop.verifier.placement = lambda *args, **kwargs: (
        times.append(loop.steps)
        or {"position": [0.4, 0.2, 0.8], "score": 0.8, "views": ["side", "front"]}
    )
    for _ in range(30):
        loop.advance(sensed)
    assert loop.done and loop.placement_verified
    assert len(times) == 3 and min(np.diff(times)) >= 9


def test_settling_waits_for_stable_images_before_spending_a_retry():
    sensed = robot()
    loop = ClosedLoopController(
        sensed, query="object", detection=detection(), detector=None, capture=None
    )
    loop.inner.phase, loop.inner.elapsed = 8, 0
    points = iter([[0.4, 0.2, 0.8], [0.42, 0.2, 0.8], [0.42, 0.2, 0.8], [0.42, 0.2, 0.8]])
    loop.verifier.placement = lambda *args, **kwargs: {
        "position": next(points),
        "views": ["side", "front"],
        "score": 0.8,
    }
    for _ in range(40):
        loop.advance(sensed)
        if loop.done:
            break
    assert loop.done and loop.attempt == 1
    assert any(event["event"] == "placement_settling" for event in loop.events)


def test_uncertain_placement_has_a_bounded_wait():
    sensed = robot()
    loop = ClosedLoopController(
        sensed, query="object", detection=detection(), detector=None, capture=None
    )
    loop.inner.phase, loop.inner.elapsed = 8, 0

    def uncertain(*args, **kwargs):
        raise DetectionRejected("Occluded")

    loop.verifier.placement = uncertain
    for _ in range(90):
        loop.advance(sensed)
        if loop.retreat is not None:
            break
    assert loop.retreat is not None and loop.attempt == 2
    assert loop.inner.elapsed <= 4.5


def test_exhausted_retry_saves_partial_recording_without_resetting(tmp_path, monkeypatch):
    assets = os.environ.get("ROBOGOGOGO_VISION_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VISION_ASSETS for native failure-recording checks")
    from robogogogo.demo.session import PandaSession

    wrong = {**detection(0.34), "image": "data:image/png;base64,", "name": "Lemon"}
    detector = SimpleNamespace(find_target=lambda *args, **kwargs: dict(wrong))
    monkeypatch.setattr("robogogogo.demo.session.PretrainedDetector", lambda _: detector)
    session = PandaSession(assets, tmp_path)
    try:
        session.call("select_object", identifier="lemon")
        before = session.call("snapshot")
        session.call("record", command="start", images=False)
        session.call("command", command="run", perception="vlm", query="Lemon")
        session.call("command", command="pause")
        for _ in range(500):
            session.call("tick")
            current = session.call("snapshot")
            if current["status"] == "failed":
                break
        assert current["episode_id"] == before["episode_id"]
        assert current["verification"]["attempt"] == 2
        assert current["verification"]["stage"] == "stopped"
        assert not current["ended"] and current["recording"] is None
        saved = current["last_saved"]
        assert saved["outcome"] == "failed"
        assert saved["end_reason"] == "verification_failed"
        assert (
            len(
                [
                    event
                    for event in saved["verification"]["events"]
                    if event["event"] == "retry_started"
                ]
            )
            == 1
        )
        assert (tmp_path / f"{saved['id']}.zip").is_file()
    finally:
        session.close()


def test_simulator_evaluation_does_not_finish_a_camera_driven_episode(monkeypatch):
    assets = os.environ.get("ROBOGOGOGO_VISION_ASSETS")
    if not assets:
        pytest.skip("Set ROBOGOGOGO_VISION_ASSETS for native completion-boundary checks")
    from robogogogo.demo.environment import PandaEnv

    with PandaEnv(assets) as env:
        env.external_termination = True
        monkeypatch.setattr(env, "_placed", lambda: True)
        for _ in range(7):
            result = env.step(env.previous_action)
        assert result[4]["evaluation_success"]
        assert not result[2] and not result[4]["success"]
        env.external_success = True
        result = env.step(env.previous_action)
        assert result[2] and result[4]["reason"] == "visually_verified"
