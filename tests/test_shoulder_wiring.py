"""Inter-motor wiring cannot pass a guard by hiding inside its own host link."""

from copy import deepcopy

import numpy as np
import pytest

from robogogogo.robot import packaging
from robogogogo.robot.mobile_design import mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.robot.shoulder_wiring import (
    branch_specs,
    cover_mass_specs,
    fixed_wiring_report,
    internal_clearance,
    internal_routes,
)
from robogogogo.robot.wiring import robot_wiring, validate_wiring
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    return loads(mobile_dual_arm_urdf()).model


def test_fixed_leads_fit_and_have_no_hidden_same_host_interference(model):
    report = fixed_wiring_report(model)
    assert report["valid"], report["failures"]
    assert len(report["leads"]) == 10
    assert all(0.09 < lead["length_m"] < 0.15 for lead in report["leads"])
    assert all(lead["minimum_gap_m"] >= 0.001 for lead in report["leads"])


def test_corrupt_fixed_lead_is_detected_even_on_same_host(model, monkeypatch):
    original = packaging.assembly_parts

    def corrupted(m):
        parts = original(m)
        part = next(p for p in parts if p["name"] == "left_waist_drive_lead_0_8")
        part["end"] = [0.0, 0.0, 0.16]  # inside its own yaw motor
        return parts

    monkeypatch.setattr(packaging, "assembly_parts", corrupted)
    assert not fixed_wiring_report(model)["valid"]


def test_branch_endpoints_and_length_follow_both_links(model):
    world = mobile_collision_world(model)
    for yaw, roll in ((-0.35, 0.4), (0.35, 0.4), (0.0, 0.0)):
        q = mobile_home(model)
        for side, sign in (("left", 1), ("right", -1)):
            q[model.indices[model.names.index(f"{side}_waist")]] = yaw * sign
            q[model.indices[model.names.index(f"{side}_upper_roll")]] = roll * sign
        pose = world._poses(q)
        routes = internal_routes(model, q, pose)
        report = internal_clearance(model, q, pose, world, routes)
        assert report.distance >= world.margin, report
        for route, spec in zip(routes, branch_specs(model)):
            points = np.array(route["points"])
            assert route["valid"]
            assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == pytest.approx(
                spec["length_m"], abs=1e-6
            )
            for endpoint, (body, anchor) in enumerate(zip(spec["bodies"], spec["anchors"])):
                expected = (
                    np.asarray(pose.position[body]) + np.asarray(pose.rotation[body]) @ anchor
                )
                np.testing.assert_allclose(points[0 if endpoint == 0 else -1], expected, atol=1e-8)


def test_cable_cannot_hide_inside_motor_or_outside_cover(model):
    world = mobile_collision_world(model)
    q = mobile_home(model)
    pose = world._poses(q)
    for local in ([0.0, 0.0, 0.16], [0.4, 0.0, 0.3]):
        routes = internal_routes(model, q, pose)
        body = model.parents[model.names.index("left_waist")]
        routes[0]["points"][32] = (
            np.asarray(pose.position[body]) + np.asarray(pose.rotation[body]) @ local
        ).tolist()
        assert internal_clearance(model, q, pose, world, routes).distance < world.margin
    q[model.indices[model.names.index("left_waist")]] = 2.0
    assert not internal_routes(model, q, world._poses(q))[0]["valid"]


def test_physical_route_bindings_and_separate_feed_throughs(model):
    wiring = robot_wiring(model)
    assert validate_wiring(wiring) == []
    yaw = next(c for c in wiring["connectors"] if c["name"] == "left_waist_panel")
    assert {"roll_V+", "pitch_V+"}.issubset(yaw["terminals"])
    damaged = deepcopy(wiring)
    next(w for w in damaged["wires"] if w.get("route_ids"))["route_ids"] = []
    assert validate_wiring(damaged)
    device = next(c for c in wiring["connectors"] if c["name"] == "left_upper_roll_drive")
    assert [p["family"] for p in device["physical_interfaces"]] == ["XT30 2P", "GH125 4P"]


def test_cover_shell_inertias_are_positive_and_added_to_model(model):
    covers = cover_mass_specs(model)
    assert sum(p["mass_kg"] for p in covers) > 0.1
    for part in covers:
        assert np.linalg.eigvalsh(part["local_inertia"]).min() > 0
    # Catalog rotor packages no longer absorb arbitrary fixture mass at their centres.
    assert float(model.mass[model.names.index("left_waist_actuator")]) == pytest.approx(0.750)


def test_export_does_not_depend_on_simulation_precision():
    from jax import enable_x64

    with enable_x64(False):
        single = mobile_dual_arm_urdf()
    with enable_x64(True):
        double = mobile_dual_arm_urdf()
    assert single == double


def test_fixed_lead_fillet_cannot_extend_behind_connector():
    from robogogogo.robot.shoulder_wiring import fixed_adapter_path

    # Middle leg has ample room, but a 60 mm fillet consumes more than the
    # 55 mm straight exit at both connectors. The old middle-only check passed.
    with pytest.raises(ValueError, match="cannot fit"):
        fixed_adapter_path([0, 0, 0], [0.11, 0.3, 0], [1, 0, 0], [1, 0, 0], radius=0.06)
    points = fixed_adapter_path([0, 0, 0], [0.3, 0, 0], [2, 0, 0], [1, 0, 0])
    assert np.isfinite(points).all()
    assert np.all(np.diff(points[:, 0]) > 0)
    with pytest.raises(ValueError):
        fixed_adapter_path([0, 0, 0], [0.3, 0, 0], [0, 0, 0], [1, 0, 0])
