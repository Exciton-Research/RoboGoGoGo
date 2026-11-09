"""Collision geometry, swept motion, detours, and coordinated task lifecycle."""

import numpy as np
import pytest

from robogogogo.planning import Capsule, CollisionWorld, PlanningError, SphereObstacle, plan_motion
from robogogogo.planning.collision import CollisionViolation, segment_distance
from robogogogo.planning.task import TaskExecutor, TaskPlan, TaskStage
from robogogogo.session import Session
from robogogogo.spatial import ModelBuilder


def xy_robot():
    builder = ModelBuilder()
    x = builder.add_link(
        "x",
        joint="prismatic",
        axis=(1, 0, 0),
        origin=(0, 0, 0.5),
        mass=1,
        inertia=np.eye(3) * 0.01,
        limits=(-1, 1),
        home=-0.8,
    )
    builder.add_link(
        "y",
        parent=x,
        joint="prismatic",
        axis=(0, 1, 0),
        mass=1,
        inertia=np.eye(3) * 0.01,
        limits=(-1, 1),
        home=0,
    )
    return builder.build()


def xy_world():
    model = xy_robot()
    return CollisionWorld(
        model,
        [Capsule("tool", 1, (0, 0, 0), (0, 0, 0), 0.05)],
        obstacles=[SphereObstacle("blocker", (0, 0, 0.5), 0.2)],
        margin=0.025,
    )


@pytest.mark.parametrize(
    "a,b,c,d,expected",
    [
        ([0, 0, 0], [1, 0, 0], [0.5, -1, 0], [0.5, 1, 0], 0),
        ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], 1),
        ([0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], 1),
        ([0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], 1),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0),
    ],
)
def test_segment_distances(a, b, c, d, expected):
    assert segment_distance(*[np.array(v, dtype=float) for v in (a, b, c, d)]) == pytest.approx(
        expected
    )


def test_swept_edge_rejects_collision_with_clear_endpoints():
    world = xy_world()
    assert world.clearance([-0.8, 0]).distance > world.margin
    assert world.clearance([0.8, 0]).distance > world.margin
    assert not world.edge_free([-0.8, 0], [0.8, 0])
    assert world.edge_free([-0.8, 0.5], [0.8, 0.5])
    with pytest.raises(PlanningError, match="goal"):
        plan_motion(world, [-0.8, 0], [0, 0])


def test_rrt_finds_reproducible_detour_and_every_edge_is_clear():
    world = xy_world()
    first = plan_motion(world, [-0.8, 0], [0.8, 0], seed=13, max_iterations=400)
    second = plan_motion(world, [-0.8, 0], [0.8, 0], seed=13, max_iterations=400)
    assert first.method == "rrt_connect"
    assert first.waypoints == second.waypoints
    assert len(first.waypoints) > 2
    for start, end in zip(first.waypoints, first.waypoints[1:]):
        assert world.edge_free(start, end)
        for alpha in np.linspace(0, 1, 101):
            assert (
                world.clearance((1 - alpha) * np.array(start) + alpha * np.array(end)).distance
                >= world.margin
            )


def test_runtime_collision_guard_does_not_commit_unsafe_substep():
    model = xy_robot()
    session = Session(model)
    # Moving rapidly toward an obstacle tests momentum, not just the desired goal.
    import jax.numpy as jnp

    from robogogogo.spatial import State

    session.state = State(jnp.array([-0.28, 0.0]), jnp.array([10.0, 0.0]))
    session.target_q = session.state.q
    before = tuple(np.asarray(x).copy() for x in session.state)
    history = len(session.history)
    with pytest.raises(CollisionViolation):
        session.advance(collision_world=xy_world())
    np.testing.assert_array_equal(session.state.q, before[0])
    np.testing.assert_array_equal(session.state.qd, before[1])
    assert len(session.history) == history and session.elapsed == 0


def test_task_waits_for_settling_and_dwell_and_can_pause():
    stage = TaskStage("inspect", "left", {"left": [0, 0, 0]}, ((0.0,), (0.2,)), "direct", 0.2)
    task = TaskExecutor(TaskPlan("test", (stage,), 0, (0, 0, 0)), max_delta=[0.1])
    task.update(np.array([0.2]), np.array([1.0]), 0, {"left": [0, 0, 0]})
    assert task.settled_since is None
    task.update(np.array([0.2]), np.array([0.0]), 0.1, {"left": [0, 0, 0]})
    task.pause()
    task.update(np.array([0.2]), np.array([0.0]), 0.4, {"left": [0, 0, 0]})
    assert task.status == "paused"
    task.resume()
    task.update(np.array([0.2]), np.array([0.0]), 0.4, {"left": [0, 0, 0]})
    assert task.status == "completed" and len(task.events) == 1


def test_supervisor_rejects_policy_edge_through_obstacle():
    from robogogogo.contracts import ActionChunk, Observation
    from robogogogo.system import ControlSystem

    class UnsafePolicy:
        def predict(self, observation):
            return ActionChunk(((1.6, 0.0),))

    world = xy_world()
    system = ControlSystem(
        {"learned": UnsafePolicy()},
        lower=[-1, -1],
        upper=[1, 1],
        max_delta=[2, 2],
        collision_world=world,
    )
    with pytest.raises(ValueError, match="collision"):
        system.step(Observation("", b"", (-0.8, 0.0), (0.0, 0.0), 0.0), controller="learned")


def test_task_action_scaling_preserves_planned_direction():
    from robogogogo.contracts import Observation

    stage = TaskStage("move", None, {}, ((0.0, 0.0), (0.2, 0.1)), "direct", 0)
    task = TaskExecutor(
        TaskPlan("test", (stage,), 0, (0, 0, 0)), max_delta=[0.1, 0.1], control_ticks=20
    )
    action = task.predict(Observation("", b"", (0.0, 0.0), (0.0, 0.0), 0))
    np.testing.assert_allclose(action.actions[0], [0.1, 0.05])
    assert action.control_ticks == 20
