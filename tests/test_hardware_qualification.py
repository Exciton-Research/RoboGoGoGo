import math

import pytest

from robogogogo.robot.hardware_qualification import candidate_bend_limit, shoulder_qualification
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
from robogogogo.urdf import loads


def test_vendor_candidates_do_not_qualify_the_custom_microloom():
    model = loads(mobile_dual_arm_urdf()).model
    report = shoulder_qualification(model)
    assert not report["hardware_ready"]
    assert len(report["branches"]) == 4
    assert all(not b["specification_compatible"] for b in report["branches"])
    assert candidate_bend_limit(0.008) == pytest.approx(0.092)
    assert candidate_bend_limit(0.015) > candidate_bend_limit(0.008)


@pytest.mark.parametrize("radius", [-1, math.nan, math.inf])
def test_invalid_bundle_size_cannot_produce_a_qualification(radius):
    with pytest.raises(ValueError):
        candidate_bend_limit(radius)
