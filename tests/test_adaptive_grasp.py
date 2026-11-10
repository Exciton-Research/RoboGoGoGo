import numpy as np
import pytest

from robogogogo.adaptive_grasp import AdaptiveGrip, GripSafety


def feed(c, speed, normal=8.0, closed=True, hardware=40.0, seconds=0.2):
    for _ in range(round(seconds / 0.01)):
        c.update(
            closed=closed,
            normals=[normal, normal],
            slip_speed=speed,
            peak_force=normal,
            hardware_limit=hardware,
            dt=0.01,
        )


def test_slip_feedback_increases_force_then_holds_recovered_level():
    c = AdaptiveGrip()
    feed(c, 0.02)
    assert c.force > 8 and c.motion_hold
    force = c.force
    feed(c, 0, seconds=0.5)
    assert c.status == "stable" and not c.motion_hold and c.force >= force
    feed(c, 0, closed=False)
    assert c.force == 8 and c.status == "idle"


def test_infeasible_grasp_faults_without_exceeding_object_command_envelope():
    c = AdaptiveGrip(GripSafety(initial_force_n=2, damage_force_n=10))
    feed(c, 0.02, normal=2, seconds=2)
    assert c.fault and c.status == "fault"
    assert c.force <= 7.5
    assert c.peak_force == 2


def test_damage_peak_is_latched_even_while_open():
    c = AdaptiveGrip()
    feed(c, 0, normal=56, closed=False)
    assert c.fault and c.force == 0 and c.peak_force == 56
    feed(c, 0, normal=0, closed=False)
    assert c.fault


def test_noise_single_contact_and_invalid_inputs_do_not_trigger_squeeze():
    c = AdaptiveGrip()
    feed(c, 0.002, seconds=1)
    assert c.force == 8
    feed(c, 0.1, normal=0, seconds=1)
    assert c.force == 8
    with pytest.raises(ValueError):
        feed(c, np.nan)
    with pytest.raises(ValueError):
        GripSafety(initial_force_n=8, damage_force_n=5)


def test_physical_friction_change_recovers_while_fixed_force_loses_grasp():
    import runpy
    from pathlib import Path

    run = runpy.run_path(str(Path(__file__).parent / "fixtures/adaptive_grasp.py"))["run"]
    adaptive, fixed = run(True), run(False)
    assert adaptive["failure"] is None
    assert adaptive["height_m"] > 0.495
    feedback = adaptive["feedback"]["arm"]
    assert feedback["status"] == "stable"
    assert 8 < feedback["force_setpoint_n"] <= 40
    assert feedback["peak_force_n"] < 55
    assert not adaptive["damage_limit_exceeded"]
    assert fixed["failure"] == "Grasp lost: arm"
    assert fixed["height_m"] < 0.48


def test_physical_unrecoverable_slip_reports_failure():
    import runpy
    from pathlib import Path

    run = runpy.run_path(str(Path(__file__).parent / "fixtures/adaptive_grasp.py"))["run"]
    result = run(True, friction=0, damage_force=12)
    assert result["failure"]
    assert result["feedback"]["arm"]["force_setpoint_n"] <= 9
    assert not result["damage_limit_exceeded"]
