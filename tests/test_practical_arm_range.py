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
        if label in ("random:16", "random:17"):
            # New upper-arm roll can point a hand toward the base in a low pose.
            # Preserve this negative case rather than exempting the collision.
            expected = (
                ("right_hand", "base_right")
                if label == "random:16"
                else ("right_hand", "left_hand")
            )
            assert rigid.pair == expected
            assert rigid.distance < world.margin
            assert not world.edge_free(mobile_home(model), q)
        else:
            assert rigid.distance >= world.margin, (label, rigid)
