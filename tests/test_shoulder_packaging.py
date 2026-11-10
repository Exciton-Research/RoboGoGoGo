"""Provisional drive volumes must be clear at home and guarded when folded."""

import numpy as np

from robogogogo.planning.collision import CollisionWorld
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.packaging import (
    arm_cable_passages,
    assembly_parts,
    packaging_capsules,
    shoulder_packages,
)
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.urdf import loads


def test_shoulder_drive_proxies_enclose_hardware_and_check_moving_upper_arms():
    model = loads(mobile_dual_arm_urdf()).model
    home = mobile_home(model)
    parts = packaging_capsules(model)
    internals = CollisionWorld(model, parts, margin=0.012)
    assert internals.clearance(home).distance > 0.012
    # Every conservative proxy must enclose the complete catalog cylinder,
    # including drive housings now mounted outside the cosmetic socket.
    for proxy, package in zip(parts, shoulder_packages(model)):
        center = np.asarray(proxy.start)
        for end in (package["start"], package["end"]):
            assert np.hypot(np.linalg.norm(np.asarray(end) - center), package["radius"]) <= proxy.radius + 1e-9
    guard = mobile_collision_world(model)
    pairs = {frozenset((guard.capsules[i].name, guard.capsules[j].name)) for i, j in guard.pairs}
    for side in ("left", "right"):
        for suffix in ("waist", "upper_roll", "shoulder"):
            assert frozenset((f"{side}_{suffix}_drive_package", f"{side}_upper_arm")) in pairs


def test_reserved_cable_bores_leave_clearance_but_do_not_claim_joint_harness():
    model = loads(mobile_dual_arm_urdf()).model
    for route in arm_cable_passages(model):
        assert route["radius"] - route["bundle_radius"] >= 0.003 - 1e-9
        assert route["end"][0] > route["start"][0]
        assert "unresolved" in route["status"]


def test_brackets_connectors_and_wrist_drives_clear_during_working_motion():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    parts = assembly_parts(model)
    assert sum(p["kind"] == "connector" for p in parts) == 14
    home = mobile_home(model)
    # Exercise all three wrist axes together, both mirrored arms, with fittings.
    for angle in np.linspace(-0.2, 0.2, 21):
        q = home.copy()
        for side in ("left", "right"):
            for suffix in ("wrist_roll", "wrist_pitch", "tool_roll"):
                q[model.indices[model.names.index(f"{side}_{suffix}")]] += angle
        report = world.clearance(q)
        assert report.distance >= world.margin, report
    # The opposite arm and remote torso remain collision pairs for wrist fittings.
    names = {frozenset((world.capsules[i].name, world.capsules[j].name)) for i, j in world.pairs}
    assert frozenset(("left_wrist_pitch_power_can_connector", "right_hand")) in names
    assert frozenset(("left_wrist_pitch_power_can_connector", "chest_middle")) in names
