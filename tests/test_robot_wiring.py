"""Electrical failures must not disappear behind an apparently complete netlist."""

from copy import deepcopy

import pytest

from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
from robogogogo.robot.wiring import robot_wiring, validate_wiring
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def wiring():
    return robot_wiring(loads(mobile_dual_arm_urdf()).model)


def test_canonical_arm_netlist(wiring):
    assert len(wiring["devices"]) == 14
    assert validate_wiring(wiring) == []
    assert wiring["qualification"]["complete_3d_branch_routing"] is False


@pytest.mark.parametrize(
    "fault", ["open", "short", "crossed_can", "id", "termination", "device", "voltage"]
)
def test_detect_electrical_faults(wiring, fault):
    broken = deepcopy(wiring)
    if fault == "open":
        broken["wires"].pop(0)
    elif fault == "short":
        broken["wires"].append(
            dict(source=["left_distribution", "48V+"], target=["left_distribution", "24V+"])
        )
    elif fault == "crossed_can":
        lead = next(w for w in broken["wires"] if w["source"] == ["left_can_gateway", "CAN_H"])
        lead["target"][1] = "CAN_L"
    elif fault == "id":
        broken["devices"][1]["can_id"] = broken["devices"][0]["can_id"]
    elif fault == "termination":
        broken["terminations"][1]["connector"] = "left_elbow_panel"
    elif fault == "device":
        broken["devices"].pop()
    else:
        broken["devices"][-1]["voltage"] = 48
    assert validate_wiring(broken)
