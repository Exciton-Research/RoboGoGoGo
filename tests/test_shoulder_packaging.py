"""Provisional drive volumes must be clear at home and guarded when folded."""

import numpy as np

from robogogogo.planning.collision import CollisionWorld
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.packaging import arm_cable_passages, assembly_parts, packaging_capsules
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.urdf import loads


def test_shoulder_drive_packages_fit_at_home_but_cannot_pass_through_each_other():
    model = loads(mobile_dual_arm_urdf()).model
    home = mobile_home(model)
    parts = packaging_capsules(model)
    internals = CollisionWorld(model, parts, margin=0.012)
    assert internals.clearance(home).distance > 0.012
    folded = home.copy()
    folded[model.indices[model.names.index("left_upper_roll")]] = np.pi / 2
    clash = internals.clearance(folded)
    assert clash.distance < 0
    assert all("drive_package" in name for name in clash.pair)
    guard = mobile_collision_world(model)
    assert not guard.edge_free(home, folded)
    for p in parts:
        if any(s in p.name for s in ("wrist", "tool_roll")):
            continue
        # Catalog cylinders use conservative enclosing spheres and enlarged shoulders.
        body = model.names.index(p.name.removesuffix("_drive_package"))
        origin = np.asarray(model.origin[body])
        assert max(
            np.linalg.norm(np.asarray(p.start) - origin),
            np.linalg.norm(np.asarray(p.end) - origin),
        ) + p.radius < (0.13 if "elbow" in p.name else 0.18)


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
