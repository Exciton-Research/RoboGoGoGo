"""Arm audit: shared shell bounds and correct actuator/output ownership."""

import numpy as np

from robogogogo.robot.arm_structure import arm_structure_specs
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.packaging import mechanical_assembly_parts, shoulder_packages
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.urdf import loads


def test_shells_and_collision_envelopes_share_dimensions():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    for spec in arm_structure_specs(model):
        name = model.names[spec["body"]]
        label = name.replace("_shoulder", "_upper_arm").replace("_elbow", "_forearm")
        capsule = next(c for c in world.capsules if c.name == label)
        assert capsule.start[0] == spec["sleeve_start"]
        assert capsule.end[0] == spec["sleeve_end"]
        assert capsule.radius == spec["radius"]
        assert (
            spec["carrier_start"]
            <= spec["sleeve_start"]
            < spec["sleeve_end"]
            <= spec["carrier_end"]
        )
    assert not any(c.name.endswith("_elbow_cap") for c in world.capsules)


def test_arm_outputs_follow_joint_children_and_keep_remote_checks():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    parts = {p["name"]: p for p in mechanical_assembly_parts(model)}
    motors = {p["name"]: p for p in shoulder_packages(model)}
    pairs = {frozenset((world.capsules[i].name, world.capsules[j].name)) for i, j in world.pairs}
    for side in ("left", "right"):
        assert len([n for n in motors if n.startswith(side + "_")]) == 7
        for suffix in ("elbow", "wrist_roll", "wrist_pitch", "tool_roll"):
            owner = f"{side}_{suffix}"
            motor = motors[owner + "_drive_package"]
            output = parts[owner + "_mount_arm_output"]
            body = model.names.index(owner)
            assert motor["body"] == model.parents[body]
            assert output["body"] == body
            q = mobile_home(model)
            before = world._poses(q.copy())
            q[model.indices[body]] += 0.2
            after = world._poses(q)
            np.testing.assert_allclose(
                before.rotation[motor["body"]], after.rotation[motor["body"]]
            )
            assert np.linalg.norm(np.asarray(before.rotation[body]) - after.rotation[body]) > 0.1
            other = "right" if side == "left" else "left"
            assert frozenset((output["name"], f"{other}_hand")) in pairs
            assert frozenset((output["name"], motor["name"])) not in pairs
