"""Practical coordinated motion keeps the real geometry guards enabled."""

import numpy as np

from robogogogo.robot.harness import harness_clearance, harness_routes
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.robot.validate_working_range import configurations
from robogogogo.urdf import loads


def test_practical_poses_and_height_transitions():
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    for label, q in configurations(model, count=24, seed=42):
        rigid = world.clearance(q)
        routes = harness_routes(model, q, world._poses(q))
        cable = harness_clearance(model, q, world._poses(q), world, routes)
        assert rigid.distance >= world.margin, (label, rigid)
        assert cable.distance >= world.margin, (label, cable)
        for route in routes:
            assert max(route["swivel_angles_rad"]) <= np.pi / 4 + 1e-9
            points = np.asarray(route["points"])
            assert abs(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() - route["length_m"]) < 1e-6
