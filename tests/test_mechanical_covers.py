import numpy as np
import pytest

from robogogogo.robot.mechanical_covers import motor_cover_specs
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
from robogogogo.robot.packaging import mechanical_assembly_parts, shoulder_packages
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.urdf import loads


def test_sleeves_fit_catalog_motors_and_existing_collision_envelopes():
    model = loads(mobile_dual_arm_urdf()).model
    covers = motor_cover_specs(model)
    assert len(covers) == 14
    for cover, motor in zip(covers, shoulder_packages(model)):
        assert cover["inner_radius"] - motor["radius"] == pytest.approx(0.001)
        assert cover["radius"] - cover["inner_radius"] == pytest.approx(0.002)
        assert cover["radius"] < motor["radius"] + motor["cover_allowance_m"]
        assert (
            np.linalg.norm(np.array(cover["start"]) - motor["start"]) < motor["cover_allowance_m"]
        )
        assert np.linalg.eigvalsh(cover["local_inertia"]).min() > 0
    assert all("_mount_" in p["name"] for p in mechanical_assembly_parts(model))
    world = mobile_collision_world(model)
    assert not any(
        any(word in c.name for word in ("harness", "connector", "lead", "hood"))
        for c in world.capsules
    )
