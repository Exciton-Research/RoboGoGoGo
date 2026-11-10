"""Catalog sizing and routed cable checks for the 1 kg/hand engineering target."""

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.robot.components import ARM_COMPONENTS, mounting_mass
from robogogogo.robot.harness import (
    MIN_BEND,
    harness_clearance,
    harness_routes,
    harness_spans,
    routed_curve,
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
    # Static rating compliance is distinct from the unmet 15% design reserve.
    for index in range(15, 29):
        assert abs(float(torque[index])) < float(model.effort[index])
    world = mobile_collision_world(model)
    assert world.clearance(q).distance >= world.margin
    assert (
        harness_clearance(model, q, sp.forward_kinematics(model, q), world).distance >= world.margin
    )


def test_dress_pack_length_and_endpoints_follow_both_attached_links(model):
    q = mobile_home(model)
    for angle in np.linspace(-0.2, 0.2, 9):
        for side in ("left", "right"):
            for suffix in ("wrist_roll", "tool_roll"):
                q[model.indices[model.names.index(f"{side}_{suffix}")]] = angle
        pose = sp.forward_kinematics(model, jnp.array(q))
        routes = harness_routes(model, q, pose)
        for route, span in zip(routes, harness_spans()):
            points = np.array(route["points"])
            assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == pytest.approx(
                span["length_m"], abs=1e-6
            )
            assert route["min_bend_m"] >= MIN_BEND
            for endpoint, (name, anchor) in enumerate(zip(span["bodies"], span["anchors"])):
                body = model.names.index(name)
                expected = np.array(pose.position[body]) + np.array(pose.rotation[body]) @ anchor
                np.testing.assert_allclose(points[0 if endpoint == 0 else -1], expected, atol=1e-8)
        world = mobile_collision_world(model)
        assert harness_clearance(model, q, pose, world, routes).distance >= world.margin
    _, _, valid = routed_curve([0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0], 0.5)
    assert not valid  # An unreachable anchor does not stretch the cable.


def test_reference_postures_keep_loops_clear_and_rotated_base_preserves_clearance(model):
    world = mobile_collision_world(model)
    for posture in ("standing", "low", "reach"):
        q = mobile_home(model, posture)
        pose = sp.forward_kinematics(model, jnp.array(q))
        report = harness_routes(model, q, pose)
        assert len(report) == 6 and all(route["valid"] for route in report)
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


def test_added_fixture_mass_is_reported_as_an_unmet_reserve_target(model):
    from robogogogo.robot.validate_harness import payload_report

    report = payload_report(model)
    assert report["within_continuous_ratings"]
    assert not report["reserve_target_met"]
    assert report["minimum_reserve"] < report["target_reserve"]


def test_clamp_contact_exemption_does_not_hide_midspan_penetration(model):
    world = mobile_collision_world(model)
    q = mobile_home(model)
    pose = world._poses(q)
    routes = harness_routes(model, q, pose)
    a, b = world.geometry(q)
    clamp = next(
        i for i, c in enumerate(world.capsules) if c.name == routes[0]["attachments"][0] + "_clamp"
    )
    routes[0]["points"][32] = ((a[clamp] + b[clamp]) / 2).tolist()
    assert harness_clearance(model, q, pose, world, routes).distance < world.margin
