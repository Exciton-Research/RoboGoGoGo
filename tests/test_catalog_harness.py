"""Catalog sizing and routed cable checks for the 1 kg/hand engineering target."""

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import spatial as sp
from robogogogo.robot.components import ARM_COMPONENTS
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
            ) == pytest.approx(spec.mass_kg)


def test_one_kg_per_hand_at_full_extension_has_static_continuous_torque_reserve(model):
    q = mobile_home(model)
    for side in ("left", "right"):
        for suffix in ARM_COMPONENTS:
            q[model.indices[model.names.index(f"{side}_{suffix}")]] = 0
    q = jnp.array(q)
    pose = sp.forward_kinematics(model, q)
    for side in ("left", "right"):
        displacement = (
            pose.position[model.names.index(f"{side}_tool")]
            - pose.position[model.names.index(f"{side}_waist")]
        )
        np.testing.assert_allclose(displacement, [0.885, 0, 0], atol=1e-8)
    torque = sp.inverse_dynamics(model, q, jnp.zeros_like(q), jnp.zeros_like(q))
    for side in ("left", "right"):
        torque += 9.81 * sp.point_jacobian(model, q, model.names.index(f"{side}_tool"))[2]
    # Static rating compliance does not qualify the deferred electrical harness.
    for index in range(15, 29):
        assert abs(float(torque[index])) < float(model.effort[index])
    world = mobile_collision_world(model)
    assert world.clearance(q).distance >= world.margin


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
    _, _, valid = routed_curve([0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0], 0.5)
    assert not valid  # An unreachable anchor does not stretch the cable.


def test_unqualified_internal_routes_fail_closed_in_every_reference_posture(model):
    world = mobile_collision_world(model)
    for posture in ("standing", "low", "reach"):
        q = mobile_home(model, posture)
        pose = sp.forward_kinematics(model, jnp.array(q))
        report = harness_routes(model, q, pose)
        assert len(report) == 10
        assert all(route["valid"] for route in report if not route.get("internal"))
        assert any(not route["valid"] for route in report if route.get("internal"))
        before = harness_clearance(model, q, pose, world)
        assert before.distance < world.margin
        assert "internal" in before.pair[0]
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


def test_horizontal_payload_reserve_does_not_inherit_tilted_home_pose(model):
    from robogogogo.robot.validate_harness import payload_report

    report = payload_report(model)
    assert report["pose"] == "horizontal_full_extension"
    assert report["within_continuous_ratings"]
    assert not report["reserve_target_met"]
    assert report["minimum_reserve"] == pytest.approx(0.078625184, abs=1e-8)
    assert report["minimum_reserve"] < report["target_reserve"]


def test_clamp_contact_exemption_does_not_hide_midspan_penetration(model, monkeypatch):
    from robogogogo.planning.collision import Capsule, Clearance, CollisionWorld
    from robogogogo.robot import shoulder_wiring

    # Isolate the external cable/clamp algorithm using explicit mathematical
    # fixtures. Production clamps were removed; current internal routing remains
    # unqualified and is checked separately above.
    monkeypatch.setattr(
        shoulder_wiring, "internal_clearance", lambda *a, **kw: Clearance(float("inf"), ("", ""))
    )
    world = CollisionWorld(
        model, [Capsule("fixture_clamp", 0, (0, 0, 1), (0, 0, 1), 0.02)], margin=0.012
    )
    q = mobile_home(model)
    pose = world._poses(q)
    routes = [
        dict(
            name=f"cable_{i}",
            points=np.c_[np.linspace(0, 1, 65), np.full(65, y), np.ones(65)].tolist(),
            valid=True,
            attachments=["fixture"] if i == 0 else [],
        )
        for i, y in enumerate((0, 2))
    ]
    assert harness_clearance(model, q, pose, world, routes).distance >= world.margin
    routes[0]["points"][32] = [0, 0, 1]
    assert harness_clearance(model, q, pose, world, routes).distance < world.margin


def test_fixed_fixture_mass_preserves_parallel_axis_inertia():
    import xml.etree.ElementTree as ET

    from robogogogo.robot.fixture_inertia import add_fixture_inertia

    xml = ET.fromstring(
        '<robot><link name="host"><inertial><mass value="2"/><origin xyz="0 0 0"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>'
    )
    add_fixture_inertia(
        xml, [dict(body_name="host", mass_kg=1.0, start=[1, -0.1, 0], end=[1, 0.1, 0], radius=0.01)]
    )
    inertial = xml.find("link/inertial")
    assert float(inertial.find("mass").get("value")) == 3.0
    np.testing.assert_allclose(
        np.fromstring(inertial.find("origin").get("xyz"), sep=" "), [1 / 3, 0, 0]
    )
    assert float(inertial.find("inertia").get("iyy")) == pytest.approx(0.1 + 0.01**2 / 2 + 2 / 3)
