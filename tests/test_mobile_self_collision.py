"""Self-collision coverage, swept certification and transactional mobile execution."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo.planning.collision import CollisionViolation
from robogogogo.robot.mobile_design import mobile_arm_definitions, mobile_dual_arm_urdf, mobile_home
from robogogogo.robot.self_collision import mobile_collision_world
from robogogogo.session import Session
from robogogogo.urdf import loads


@pytest.fixture(scope="module")
def model():
    m = loads(mobile_dual_arm_urdf()).model
    return replace(m, home=jnp.asarray(mobile_home(m)))


@pytest.fixture(scope="module")
def world(model):
    return mobile_collision_world(model)


def configuration(model, values):
    q = np.asarray(model.home).copy()
    q[15:29] = values
    return q


TORSO = [
    1.23334888,
    1.53483357,
    0.08175997,
    -1.82517084,
    -0.625853,
    0.31528594,
    0.44626292,
    0.92901622,
    0.51630636,
    -0.1465064,
    -1.99937653,
    -1.40590484,
    1.14234858,
    -0.60396918,
]
CROSS = [
    2.4374624,
    -0.66638913,
    1.5006716,
    -1.18953215,
    -2.15489832,
    0.02869383,
    -0.41087644,
    -0.98319743,
    0.48366767,
    0.11745401,
    -1.83026804,
    1.52780982,
    1.16189549,
    -0.48715728,
]
# Seed 187 counterexamples for engineering-11. Both endpoints clear the current
# catalog geometry, while their connecting midpoint intersects the chest.
EDGE_A = [
    1.008922,
    -0.08748693,
    0.24787374,
    -1.12485508,
    1.96358317,
    0.1427406,
    0.12095307,
    2.00748693,
    -0.3558781,
    -0.55353563,
    -1.99733075,
    0.23793184,
    -0.58318993,
    0.5367538,
]
EDGE_B = [
    0.70949583,
    -0.00517599,
    0.29568379,
    -1.70863222,
    -1.50105079,
    0.53271941,
    -0.1834423,
    1.67761876,
    -0.35821479,
    -1.73356118,
    -1.54756059,
    -1.12148512,
    -1.14998643,
    -0.17859157,
]


def test_reference_postures_and_height_paths_have_clearance(model, world):
    for posture in ("standing", "low", "reach"):
        q = mobile_home(model, posture)
        assert world.clearance(q).distance > world.margin
        assert world.edge_free(model.home, q)
    # Continuous wheel spins do not invalidate collision bounds; shared base
    # translation/yaw preserves self-clearance on this flat plane.
    moved = np.asarray(model.home).copy()
    moved[:3] = [0.5, 0.3, 0.7]
    for name in ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"):
        moved[model.indices[model.names.index(name)]] = 1000
    assert world.clearance(moved).distance == pytest.approx(
        world.clearance(model.home).distance, abs=1e-6
    )


@pytest.mark.parametrize("values,kind", [(TORSO, "torso"), (CROSS, "cross")])
def test_colliding_arm_configurations_are_detected(model, world, values, kind):
    goal = configuration(model, values)
    clearance = world.clearance(goal)
    assert clearance.distance < -0.02
    if kind == "torso":
        assert any(name.startswith("chest") for name in clearance.pair)
    else:
        assert any(name.startswith("left_") for name in clearance.pair)
        assert any(name.startswith("right_") for name in clearance.pair)
    assert not world.edge_free(model.home, goal)


def test_clear_endpoints_cannot_tunnel_through_torso(model, world):
    a, b = configuration(model, EDGE_A), configuration(model, EDGE_B)
    assert world.clearance(a).distance > world.margin
    assert world.clearance(b).distance > world.margin
    assert world.clearance((a + b) / 2).distance < 0
    assert not world.edge_free(a, b)
    assert world.last_edge_clearance.distance < world.margin
    assert not world.edge_free(a, b, max_depth=0)


def test_every_opposite_arm_and_arm_body_pair_remains_enabled(world):
    pairs = {frozenset((world.capsules[i].name, world.capsules[j].name)) for i, j in world.pairs}
    for left in (
        "upper_arm",
        "forearm",
        "elbow_drive_package",
        "wrist_roll",
        "wrist_pitch",
        "wrist_tip",
        "hand",
    ):
        for right in (
            "upper_arm",
            "forearm",
            "elbow_drive_package",
            "wrist_roll",
            "wrist_pitch",
            "wrist_tip",
            "hand",
        ):
            assert frozenset((f"left_{left}", f"right_{right}")) in pairs
        assert frozenset((f"left_{left}", "chest_upper")) in pairs
        assert frozenset((f"left_{left}", "lower_torso")) in pairs
        assert frozenset((f"left_{left}", "base_left")) in pairs


def test_command_rejection_safe_motion_and_substep_rollback(model):
    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    goal_before = np.asarray(session.target_q).copy()
    with pytest.raises(ValueError, match="Self-collision"):
        session.set_joints(configuration(model, TORSO))
    np.testing.assert_array_equal(session.target_q, goal_before)
    session.set_joints(mobile_home(model, "low"))
    with pytest.raises(ValueError, match="Finish or cancel"):
        session.set_drive([0.05, 0, 0.05], duration=0.3)
    for _ in range(1000):
        session.advance()
        assert session.mobile_status["collision_guard"] == "clear"
        assert (
            session.mobile_status["collision_clearance_m"] >= session.mobile.collision_world.margin
        )
    torso_index = model.indices[model.names.index("torso_upper")]
    assert abs(float(session.state.q[torso_index]) + 2.1) < 0.05
    before, elapsed, count = session.state, session.elapsed, len(session.history)
    _, diagnostics = session.mobile.tire_forces(before, 0.7, jnp.zeros(3))
    # A clear final state must not hide an unsafe intermediate physics substep.
    session.mobile.predict = lambda *args: (
        before,
        jnp.zeros(3),
        0.2,
        jnp.zeros_like(before.q),
        diagnostics,
        jnp.asarray([configuration(model, CROSS), before.q]),
    )
    session.cancel_motion_plan()
    session.set_drive([0.1, 0, 0])
    with pytest.raises(CollisionViolation, match="Self-collision"):
        session.advance()
    assert session.state is before
    assert session.elapsed == elapsed and len(session.history) == count
    np.testing.assert_array_equal(session.drive_twist, 0)
    assert session.mobile_status["collision_guard"] == "blocked"


def test_chest_envelopes_cover_the_rendered_shell(world):
    from robogogogo.planning.collision import segment_distance

    points = np.array(
        [
            [
                0.15 * np.sin(theta) * np.cos(phi),
                0.26 * np.sin(theta) * np.sin(phi),
                0.14 + 0.19 * np.cos(theta),
            ]
            for theta in np.linspace(0, np.pi, 61)
            for phi in np.linspace(0, 2 * np.pi, 101)
        ]
    )
    distances = []
    for capsule in world.capsules:
        if capsule.name.startswith("chest_"):
            distances.append(
                segment_distance(
                    points,
                    points,
                    np.broadcast_to(capsule.start, points.shape),
                    np.broadcast_to(capsule.end, points.shape),
                )
                - capsule.radius
            )
    assert np.max(np.min(distances, axis=0)) <= 0


def test_invalid_joint_prediction_keeps_fault_telemetry_json_serializable(model):
    import json

    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    before = session.state
    invalid = before.q.at[model.indices[model.names.index("torso_upper")]].set(-2.4)
    _, diagnostics = session.mobile.tire_forces(before, 0.7, jnp.zeros(3))
    session.mobile.predict = lambda *args: (
        before,
        jnp.zeros(3),
        0.2,
        jnp.zeros_like(before.q),
        diagnostics,
        jnp.asarray([invalid]),
    )
    with pytest.raises(CollisionViolation):
        session.advance()
    assert session.mobile_status["collision_clearance_m"] is None
    assert "joint_limits" in session.mobile_status["collision_pair"]
    json.dumps(session.mobile_status, allow_nan=False)
    assert session.state is before
