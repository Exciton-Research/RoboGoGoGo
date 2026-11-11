"""Provisional drive volumes must be clear at home and guarded when folded."""

import numpy as np

from robogogogo.planning.collision import CollisionWorld
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.packaging import (
    arm_cable_passages,
    mechanical_assembly_parts,
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
            assert (
                np.hypot(
                    np.linalg.norm(np.asarray(end) - center),
                    package["radius"] + package["cover_allowance_m"],
                )
                <= proxy.radius + 1e-9
            )
    guard = mobile_collision_world(model)
    folded = home.copy()
    folded[model.indices[model.names.index("left_upper_roll")]] = -1.5
    assert guard.clearance(folded).distance < 0
    assert not guard.edge_free(home, folded)
    pairs = {frozenset((guard.capsules[i].name, guard.capsules[j].name)) for i, j in guard.pairs}
    for side in ("left", "right"):
        for suffix in ("waist", "upper_roll"):
            assert frozenset((f"{side}_{suffix}_drive_package", f"{side}_upper_arm")) in pairs


def test_reserved_cable_bores_leave_clearance_but_do_not_claim_joint_harness():
    model = loads(mobile_dual_arm_urdf()).model
    for route in arm_cable_passages(model):
        assert route["radius"] - route["bundle_radius"] >= 0.003 - 1e-9
        assert route["end"][0] > route["start"][0]
        assert "unresolved" in route["status"]


def test_motor_mounts_and_wrist_drives_clear_during_working_motion():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    parts = mechanical_assembly_parts(model)
    assert len({p["owner"] for p in parts}) == 14
    assert all(p["kind"] == "bracket" for p in parts)
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
    assert frozenset(("left_wrist_pitch_mount_rail_1", "right_hand")) in names
    assert frozenset(("left_wrist_pitch_mount_rail_1", "chest_middle")) in names


def test_each_shoulder_contains_exactly_three_catalog_motors():
    model = loads(mobile_dual_arm_urdf()).model
    packages = shoulder_packages(model)
    for side in ("left", "right"):
        motors = [p for p in packages if p["name"].startswith(side + "_") and p["shoulder_axis"]]
        assert {p["shoulder_axis"] for p in motors} == {"lift", "roll", "swing"}
        assert len(motors) == 3
        assert all(p["cover_allowance_m"] == 0.004 for p in motors)


def test_compact_shoulder_uses_lift_then_axial_upper_arm_rotation():
    model = loads(mobile_dual_arm_urdf()).model
    for side in ("left", "right"):
        for suffix, axis in (
            ("waist", [0, 1, 0]),
            ("upper_roll", [0, 0, 1]),
            ("shoulder", [1, 0, 0]),
        ):
            np.testing.assert_allclose(model.axis[model.names.index(f"{side}_{suffix}")], axis)
    q = mobile_home(model)
    world = mobile_collision_world(model)
    before = world._poses(q.copy())
    q[model.indices[model.names.index("left_shoulder")]] += 0.3
    after = world._poses(q)
    # Axial rotation must not displace the elbow pivot, but rotates the forearm plane.
    elbow = model.names.index("left_elbow")
    np.testing.assert_allclose(before.position[elbow], after.position[elbow], atol=1e-7)
    assert np.linalg.norm(np.asarray(before.rotation[elbow]) - after.rotation[elbow]) > 0.1


def test_shoulder_fork_carriers_leave_clearance_around_motor_covers():
    model = loads(mobile_dual_arm_urdf()).model
    parts = mechanical_assembly_parts(model)
    for motor in shoulder_packages(model):
        if not motor["shoulder_axis"]:
            continue
        owner = motor["name"].removesuffix("_drive_package")
        fork = [
            part
            for part in parts
            if part["owner"] == owner and part.get("render_style") == "shoulder_yoke"
        ]
        assert len(fork) == 3
        assert all(part["render_style"] == "shoulder_yoke" for part in fork)
        # Rectangular cheek mesh stays inside the same capsule used by the guard.
        for part in fork:
            assert np.linalg.norm(np.asarray(part["section_m"]) / 2) <= part["radius"]
        assert 0.014 - fork[0]["radius"] - 0.003 > 0.004


def test_lift_motors_are_inboard_and_cross_axis_motors_follow_lift():
    model = loads(mobile_dual_arm_urdf()).model
    motors = {p["name"]: p for p in shoulder_packages(model)}
    for side, sign in (("left", 1), ("right", -1)):
        lift = motors[f"{side}_waist_drive_package"]
        swing = motors[f"{side}_upper_roll_drive_package"]
        assert lift["body"] == model.names.index(f"{side}_pedestal")
        assert swing["body"] == model.names.index(f"{side}_waist")
        # Lift is on the torso side of the joint, never a vertical chimney.
        assert all(point[1] * sign < -0.1 for point in (lift["start"], lift["end"]))
        assert all(abs(point[2]) < 1e-8 for point in (lift["start"], lift["end"]))
        assert swing["start"][2] < 0 < swing["end"][2]


def test_cross_axis_motor_is_centered_on_support_junction():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    q = mobile_home(model)
    poses = world._poses(q)
    for side in ("left", "right"):
        lift_body = model.names.index(f"{side}_waist")
        centers = {}
        for motor in shoulder_packages(model):
            if not motor["name"].startswith(side + "_") or not motor["shoulder_axis"]:
                continue
            local = (np.asarray(motor["start"]) + motor["end"]) / 2
            world_center = poses.position[motor["body"]] + poses.rotation[motor["body"]] @ local
            centers[motor["shoulder_axis"]] = poses.rotation[lift_body].T @ (
                world_center - poses.position[lift_body]
            )
        # The lift-frame XY plane contains all three centers at neutral swing.
        assert all(abs(center[2]) < 1e-7 for center in centers.values())
        np.testing.assert_allclose(centers["swing"], [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(centers["roll"], [0.14095, 0, 0], atol=1e-7)
        assert abs(abs(centers["lift"][1]) - 0.15095) < 1e-7


def test_distal_sleeve_collision_envelope_covers_its_full_length():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    for side in ("left", "right"):
        sleeve = next(c for c in world.capsules if c.name == f"{side}_upper_arm")
        assert sleeve.start == (0.202, 0, 0)
        assert sleeve.end == (0.255, 0, 0)
        assert sleeve.radius >= 0.055
        # The proximal drive remains checked against the centered cross-axis drive.
        pairs = {
            frozenset((world.capsules[i].name, world.capsules[j].name)) for i, j in world.pairs
        }
        assert (
            frozenset((f"{side}_upper_roll_drive_package", f"{side}_shoulder_drive_package"))
            in pairs
        )


def test_serial_connections_follow_each_motor_output_and_preserve_remote_guards():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    parts = mechanical_assembly_parts(model)
    pairs = {frozenset((world.capsules[i].name, world.capsules[j].name)) for i, j in world.pairs}
    for side in ("left", "right"):
        expected = {
            "torso_to_lift": f"{side}_pedestal",
            "lift_to_swing": f"{side}_waist",
            "swing_to_roll": f"{side}_upper_roll",
            "roll_to_arm": f"{side}_shoulder",
        }
        connections = [
            p
            for p in parts
            if p["owner"].startswith(side + "_") and p.get("connection_role") in expected
        ]
        assert {p["connection_role"] for p in connections} == set(expected)
        for p in connections:
            assert model.names[p["body"]] == expected[p["connection_role"]]
            assert p["mass_kg"] > 0
        shaft = f"{side}_upper_roll_mount_swing_output"
        assert frozenset((shaft, f"{side}_upper_roll_drive_package")) not in pairs
        opposite = "right" if side == "left" else "left"
        assert frozenset((shaft, f"{opposite}_hand")) in pairs
