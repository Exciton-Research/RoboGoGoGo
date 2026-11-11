"""Practical coordinated motion keeps the real geometry guards enabled."""

from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.robot.validate_working_range import configurations
from robogogogo.urdf import loads


def test_practical_poses_and_height_transitions():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    for label, q in configurations(model, count=24, seed=42):
        rigid = world.clearance(q)
        # The rearward swing pivot exposes inward-folding interference. Keep
        # these negative cases guarded instead of exempting arm/chest or mounts.
        blocked = {
            "random:1": ("right_upper_arm", "chest_upper"),
            "random:6": ("left_shoulder_mount_cheek_-1", "left_waist_drive_package"),
            "random:9": ("left_upper_arm", "chest_upper"),
            "random:13": ("right_upper_arm", "chest_upper"),
            "random:15": ("right_shoulder_mount_cheek_1", "right_waist_drive_package"),
            "random:16": ("right_upper_arm", "chest_upper"),
            "random:17": ("left_shoulder_mount_cheek_-1", "left_waist_mount_cheek_1"),
            "random:19": ("left_upper_arm", "chest_upper"),
            "random:20": ("left_shoulder_mount_cheek_-1", "left_waist_mount_cheek_1"),
            "random:22": ("left_upper_arm", "chest_upper"),
        }
        if label in blocked:
            assert rigid.pair == blocked[label]
            assert rigid.distance < world.margin
            assert not world.edge_free(mobile_home(model), q)
        else:
            assert rigid.distance >= world.margin, (label, rigid)
