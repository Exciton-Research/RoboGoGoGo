"""Practical coordinated motion keeps the real geometry guards enabled."""

from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.robot.validate_working_range import configurations
from robogogogo.urdf import loads


def test_practical_poses_and_height_transitions():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    for label, q in configurations(model, count=24, seed=42):
        rigid = world.clearance(q)
        assert rigid.distance >= world.margin, (label, rigid)
