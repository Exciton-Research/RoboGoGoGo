"""Catalog sizing and routed cable checks for the 1 kg/hand engineering target."""

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.robot.components import ARM_COMPONENTS, mounting_mass
from robogogogo.robot.harness import (
    FREE_LENGTH,
    MIN_BEND,
    harness_clearance,
    harness_routes,
    loop_points,
)
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    return loads(mobile_dual_arm_urdf()).model


def test_catalog_masses_and_continuous_ratings_are_in_dynamics(model):
    for side in ("left", "right"):
        for suffix in ARM_COMPONENTS:
            spec = ARM_COMPONENTS[suffix]
            body = model.names.index(f"{side}_{suffix}")
            assert float(model.effort[model.indices[body]]) == spec.rated_nm
            assert float(
                model.mass[model.names.index(f"{side}_{suffix}_actuator")]
            ) == pytest.approx(spec.mass_kg + mounting_mass(suffix))


def test_one_kg_per_hand_at_full_extension_has_static_continuous_torque_reserve(model):
    q = mobile_home(model)
    for side in ("left", "right"):
        for suffix in ("shoulder", "elbow", "wrist_pitch"):
            q[model.indices[model.names.index(f"{side}_{suffix}")]] = 0
    q = jnp.array(q)
    torque = sp.inverse_dynamics(model, q, jnp.zeros_like(q), jnp.zeros_like(q))
    for side in ("left", "right"):
        torque += 9.81 * sp.point_jacobian(model, q, model.names.index(f"{side}_tool"))[2]
    # 15% static reserve with integrated wrist and saddle mass; no payload acceleration or fatigue certification.
    for index in range(15, 29):
        assert abs(float(torque[index])) < 0.85 * float(model.effort[index])
    world = mobile_collision_world(model)
    assert world.clearance(q).distance >= world.margin
    assert (
        harness_clearance(model, q, sp.forward_kinematics(model, q), world).distance >= world.margin
    )


def test_service_loops_keep_length_and_report_excess_bend():
    for angle in np.linspace(-0.7, 0.7, 15):
        points, bend = loop_points([0, 1, 0], angle)
        assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == pytest.approx(
            FREE_LENGTH, abs=1e-6
        )
        assert bend >= MIN_BEND
        assert np.linalg.norm(points[0] - points[-1]) > 0.1
    assert loop_points([0, 1, 0], 2.5)[1] < MIN_BEND


def test_reference_postures_keep_loops_clear_and_rotated_base_preserves_clearance(model):
    world = mobile_collision_world(model)
    for posture in ("standing", "low", "reach"):
        q = mobile_home(model, posture)
        pose = sp.forward_kinematics(model, jnp.array(q))
        report = harness_routes(model, q, pose)
        assert len(report) == 8 and all(route["valid"] for route in report)
        before = harness_clearance(model, q, pose, world)
        assert before.distance >= world.margin
        q[:3] = [0.4, -0.3, 0.6]
        moved = harness_clearance(model, q, sp.forward_kinematics(model, jnp.array(q)), world)
        assert moved.distance == pytest.approx(before.distance, abs=1e-6)


def test_external_harness_checks_floor(model):
    world = mobile_collision_world(model)
    q = mobile_home(model)
    pose = sp.forward_kinematics(model, jnp.array(q))
    routes = harness_routes(model, q, pose)
    world.floor = 10.0
    report = harness_clearance(model, q, pose, world, routes)
    assert report.distance < 0
    assert report.pair[1] == "floor"
