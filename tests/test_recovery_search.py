"""Recovery route feasibility, narrow contact permissions, and physical execution."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robogogogo.planning import SphereObstacle
from robogogogo.planning.motion import PlanningError
from robogogogo.recovery_planning import RecoveryPlanner, contact_world

spec = importlib.util.spec_from_file_location(
    "recovery_search", Path(__file__).parents[1] / "examples/recovery_search.py"
)
BENCH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BENCH)


def setup_planner(extra_obstacles=()):
    session, grasp, world = BENCH.fixture(extra_obstacles)
    grasp.grippers["left"]["holding"] = True
    return session, grasp, world, RecoveryPlanner(session.model, grasp.arms)


def test_helper_search_keeps_retainer_fixed_and_checks_all_edges():
    session, grasp, world, planner = setup_planner()
    route = planner.helper_path(grasp, session, world, "left")
    assert route.helper == "right"
    assert len(route.joints) > 2
    assert route.contact_start > 1
    assert planner.checked >= len(route.joints) - 1
    for q in route.joints:
        np.testing.assert_array_equal(q[:3], session.state.q[:3])
    for i, (a, b) in enumerate(zip(route.joints, route.joints[1:]), 1):
        oracle = contact_world(world, grasp, "right" if i >= route.contact_start else None)
        assert oracle.edge_free(
            a, b, object_start=grasp.object_segment(), object_end=grasp.object_segment()
        )
    assert np.linalg.norm(route.positions[-1][1] - route.positions[0][1]) > 0.1


def test_contact_permission_never_exempts_palm_or_other_obstacles():
    session, grasp, world, _ = setup_planner()
    permitted = contact_world(world, grasp, "right", 1)
    names = {tuple(world.capsules[i].name for i in pair) for pair in permitted.pairs}
    assert any("right:palm" in pair and "payload" in pair for pair in names)
    assert not any(
        any(n.startswith("right:finger:") for n in pair) and "payload" in pair for pair in names
    )
    assert ("payload", "support_1") not in world.support_contacts
    assert ("payload", "support_1") in permitted.support_contacts
    assert (
        not permitted.obstacle_exclusions
        - world.obstacle_exclusions
        - {("right:finger:-1", "transfer_bar"), ("right:finger:1", "transfer_bar")}
    )


def test_blocked_helper_search_rejects_without_moving_or_opening():
    session, grasp, world, planner = setup_planner(
        [SphereObstacle("blocked_helper", (0.2, 0.25, 0.8), 0.08)]
    )
    before = np.asarray(session.state.q).copy()
    with pytest.raises(PlanningError):
        planner.helper_path(grasp, session, world, "left")
    np.testing.assert_array_equal(session.state.q, before)
    assert grasp.grippers["left"]["command"] == "closed"
    assert planner.rejections


def test_search_requires_collision_geometry():
    session, grasp, _, planner = setup_planner()
    with pytest.raises(PlanningError, match="collision geometry"):
        planner.support_path(grasp, session, None)


@pytest.mark.parametrize("scenario,phase", [("helper", "secured"), ("support", "placed")])
def test_physical_recovery_search(scenario, phase):
    result = BENCH.run(scenario)
    assert result["failure"] is None
    assert result["recovery"]["phase"] == phase
    assert result["recovery"]["search_checks"] > 0
    assert not result["object"]["material"]["fractured"]
    events = {event["event"] for event in result["events"]}
    if scenario == "helper":
        assert "recovery_approaching_helper" in events
        assert all(g["holding"] for g in result["grippers"].values())
        assert not result["object"]["material"]["damaged"]
    else:
        assert "recovery_transporting" in events
        assert result["object"]["material"]["damage_fraction"] >= 0.02
        support = result["recovery"]["support_index"]
        assert result["support_load_n"][support + 1] > 0.8 * 0.16 * 9.81
        assert all(g["command"] == "open" for g in result["grippers"].values())


def test_support_search_falls_back_when_nearest_destination_is_obstructed():
    session, grasp, world, planner = setup_planner(
        [SphereObstacle("occupied_support", (0.48, 0.0, 0.52), 0.045)]
    )
    offset = np.array([0.15, 0.0, 0.10])
    session.state = session.state._replace(q=session.state.q.at[:3].set(offset))
    grasp.state = grasp.state._replace(position=grasp.state.position + offset)
    route = planner.support_path(grasp, session, world)
    assert route.support == 0
    assert any(r.get("support") == 1 for r in planner.rejections)
    np.testing.assert_allclose(route.centers[-1][:2], grasp.supports[0, :2], atol=0.002)
    assert np.linalg.norm(route.centers[-1][:2] - route.centers[0][:2]) > 0.1


def test_old_support_plane_does_not_block_a_lower_distant_placement():
    session, grasp, world, _ = setup_planner()
    offset = np.array([0.28, 0.0, -0.03])
    session.state = session.state._replace(q=session.state.q.at[:3].set(offset))
    grasp.state = grasp.state._replace(position=grasp.state.position + offset)
    oracle = contact_world(world, grasp, support=1)
    clearance = oracle.clearance(session.state.q, grasp.object_segment())
    assert clearance.distance >= world.margin


def test_movement_rejects_stale_object_transform_without_releasing():
    session, grasp, world, planner = setup_planner()
    grasp.enable_recovery()
    recovery = grasp.recovery
    recovery.route = planner.helper_path(grasp, session, world, "left")
    recovery.phase, recovery.helper = "approaching_helper", "right"
    grasp.state = grasp.state._replace(position=grasp.state.position + np.array([0.02, 0, 0]))
    recovery.prepare(grasp, session, world)
    assert recovery.phase == "blocked"
    assert "planned recovery envelope" in grasp.failure
    assert grasp.grippers["left"]["command"] == "closed"


def test_material_risk_pauses_route_and_new_damage_blocks_it(monkeypatch):
    session, grasp, world, planner = setup_planner()
    grasp.enable_recovery()
    recovery = grasp.recovery
    recovery.route = planner.helper_path(grasp, session, world, "left")
    recovery.phase, recovery.helper = "approaching_helper", "right"
    material = dict(risk=0.6, damage_fraction=0.0)
    monkeypatch.setattr(grasp, "material_status", lambda: material)
    recovery.prepare(grasp, session, world)
    assert recovery.risk_hold
    assert recovery.route_alpha == 0
    np.testing.assert_array_equal(session.target_q, session.state.q)
    material["risk"] = 0.4
    recovery.prepare(grasp, session, world)
    assert not recovery.risk_hold
    assert recovery.route_alpha > 0
    material["damage_fraction"] = 0.02
    recovery.prepare(grasp, session, world)
    assert recovery.phase == "blocked"
    assert "damage increased" in grasp.failure
    assert grasp.grippers["left"]["command"] == "closed"


def test_bimanual_transport_preserves_tool_separation_after_helper_recruitment():
    session, grasp, world, planner = setup_planner()
    approach = planner.helper_path(grasp, session, world, "left")
    session.state = session.state._replace(q=session.state.q.at[:].set(approach.joints[-1]))
    grasp.grippers["right"]["holding"] = True
    # The base task world still only permits left-finger contact.
    pose = np.asarray(session.poses().position)[list(planner.bodies)]
    target = pose + np.array([0.0, 0.0, 0.05])
    route = planner.path(grasp, session, world, [target], [True, True])
    separation = pose[1] - pose[0]
    for positions in route.positions:
        np.testing.assert_allclose(positions[1] - positions[0], separation, atol=1e-6)
    np.testing.assert_allclose(route.centers[-1] - route.centers[0], [0, 0, 0.05], atol=0.002)
