"""Explicit task stages, shared-workspace ownership, and settled-state completion."""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..contracts import ActionChunk
from ..control import inverse_kinematics_targets
from .motion import PlanningError, plan_motion


@dataclass(frozen=True)
class TaskStage:
    name: str
    owner: str | None
    targets: dict[str, list[float]]
    path: tuple[tuple[float, ...], ...]
    method: str
    dwell_seconds: float = 0.2
    operation: str | None = None
    rotations: dict[str, list[list[float]]] | None = None


@dataclass(frozen=True)
class TaskPlan:
    name: str
    stages: tuple[TaskStage, ...]
    seed: int
    station: tuple[float, float, float]
    kind: str = "inspection"

    def to_dict(self):
        return asdict(self)


def plan_shared_inspection(model, arms, world, initial_q, *, station=(0.55, 0, 0.75), seed=7):
    """Two arms inspect one station in order, then return to their initial poses.

    Both arms reposition together during staging. The station is reserved by one
    arm at a time; the other has a parking goal 25 cm to its own side. This is a
    position-only inspection fixture, not grasping, payload transport or handover.
    """
    if len(arms) != 2 or set(arms[0].joint_indices) & set(arms[1].joint_indices):
        raise ValueError("Shared inspection requires two independent arm groups")
    station = np.asarray(station, dtype=float)
    if station.shape != (3,) or not np.isfinite(station).all():
        raise ValueError("Station must contain three finite coordinates")
    bodies = tuple(a.body for a in arms)
    from .. import spatial as sp

    initial_q = np.asarray(initial_q, dtype=float)
    initial_positions = np.asarray(sp.forward_kinematics(model, jnp.asarray(initial_q)).position)
    home = {a.name: initial_positions[a.body].tolist() for a in arms}
    # Determine physical left/right from each arm's root, not arbitrary names.
    roots = []
    for arm in arms:
        body = arm.body
        while model.parents[body] >= 0 and model.indices[model.parents[body]] >= 0:
            body = model.parents[body]
        roots.append(initial_positions[body, 1])
    order = np.argsort(roots)[::-1]
    left, right = arms[int(order[0])], arms[int(order[1])]
    parked = {
        left.name: (station + [0, 0.25, 0.06]).tolist(),
        right.name: (station + [0, -0.25, 0.06]).tolist(),
    }
    specifications = [
        ("Stage both arms", None, parked),
        ("Left arm inspects", left.name, {**parked, left.name: station.tolist()}),
        ("Clear shared station", None, parked),
        ("Right arm inspects", right.name, {**parked, right.name: station.tolist()}),
        ("Return both arms", None, home),
    ]
    solve = jax.jit(lambda q, t: inverse_kinematics_targets(model, t, bodies, q, iterations=120))
    stages = []
    current = initial_q
    for index, (name, owner, targets) in enumerate(specifications):
        goal, residual = solve(jnp.asarray(current), jnp.asarray([targets[a.name] for a in arms]))
        if not np.isfinite(goal).all() or np.max(residual) > 0.005:
            raise PlanningError(
                f"{name}: target unreachable (IK error {float(np.max(residual)):.4f} m)"
            )
        motion = plan_motion(world, current, np.asarray(goal), seed=seed + index)
        stages.append(TaskStage(name, owner, targets, motion.waypoints, motion.method))
        current = np.asarray(goal)
    return TaskPlan("Shared-station inspection", tuple(stages), seed, tuple(station))


class TaskExecutor:
    """Feedback-driven waypoint execution with explicit pause, failure and timeout."""

    def __init__(
        self,
        plan,
        *,
        max_delta,
        started_at=0.0,
        timeout=60.0,
        joint_tolerance=0.015,
        velocity_tolerance=0.06,
        control_ticks=10,
    ):
        if (
            not np.isfinite([joint_tolerance, velocity_tolerance]).all()
            or min(joint_tolerance, velocity_tolerance) <= 0
        ):
            raise ValueError("Task tolerances must be finite and positive")
        self.joint_tolerance, self.velocity_tolerance = joint_tolerance, velocity_tolerance
        if not isinstance(control_ticks, int) or not 1 <= control_ticks <= 20:
            raise ValueError("Control ticks must be in [1, 20]")
        self.control_ticks = control_ticks
        self.plan = plan
        self.limit = np.asarray(max_delta, dtype=float)
        if self.limit.ndim != 1 or (self.limit <= 0).any() or not np.isfinite(self.limit).all():
            raise ValueError("Invalid task action limits")
        if not plan.stages or not np.isfinite([started_at, timeout]).all() or timeout <= 0:
            raise ValueError("Invalid task plan or timeout")
        for stage in plan.stages:
            path = np.asarray(stage.path)
            if (
                path.ndim != 2
                or path.shape[1:] != self.limit.shape
                or len(path) < 2
                or not np.isfinite(path).all()
                or not np.isfinite(stage.dwell_seconds)
                or stage.dwell_seconds < 0
            ):
                raise ValueError("Invalid task stage path")
        self.stage_index = 0
        self.waypoint = 1
        self.status = "ready"
        self.reason = None
        self.settled_since = None
        self.started_at = started_at
        self.timeout = timeout
        self.events = []
        self.minimum_clearance = None

    @property
    def stage(self):
        return self.plan.stages[min(self.stage_index, len(self.plan.stages) - 1)]

    def block(self, reason):
        self.status = "blocked"
        self.reason = reason

    def pause(self):
        if self.status in ("ready", "running"):
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self.status = "ready"

    def update(self, q, qd, time, tool_positions, *, tool_rotations=None, completion_ready=True):
        if self.status not in ("ready", "running"):
            return
        if time - self.started_at > self.timeout:
            self.block("Task simulation-time budget exceeded")
            return
        goal = np.asarray(self.stage.path[self.waypoint])
        settled = (
            np.max(np.abs(goal - q)) < self.joint_tolerance
            and np.max(np.abs(qd)) < self.velocity_tolerance
        )
        if not settled:
            self.settled_since = None
            return
        if self.waypoint < len(self.stage.path) - 1:
            self.waypoint += 1
            return
        if any(
            np.linalg.norm(np.asarray(tool_positions[name]) - target) > 0.015
            for name, target in self.stage.targets.items()
        ):
            self.settled_since = None
            return
        if self.stage.rotations:
            if tool_rotations is None:
                raise ValueError("Pose task requires tool rotation feedback")
            for name, desired in self.stage.rotations.items():
                actual = np.asarray(tool_rotations[name])
                if actual.shape != (3, 3) or not np.isfinite(actual).all():
                    raise ValueError("Invalid tool rotation feedback")
                angle = np.arccos(
                    np.clip((np.trace(np.asarray(desired).T @ actual) - 1) / 2, -1, 1)
                )
                if angle > 0.04:
                    self.settled_since = None
                    return
        if not completion_ready:
            self.settled_since = None
            return
        if self.settled_since is None:
            self.settled_since = time
        if time - self.settled_since + 1e-9 < self.stage.dwell_seconds:
            return
        self.events.append(
            dict(
                stage=self.stage_index,
                name=self.stage.name,
                owner=self.stage.owner,
                completed_at=time,
            )
        )
        self.stage_index += 1
        self.waypoint = 1
        self.settled_since = None
        if self.stage_index == len(self.plan.stages):
            self.status = "completed"

    def predict(self, observation):
        if self.status not in ("ready", "running"):
            raise ValueError(f"Task is {self.status}")
        self.status = "running"
        goal = np.asarray(self.stage.path[self.waypoint])
        delta = goal - np.asarray(observation.joint_position)
        scale = min(1.0, float(np.min(self.limit / np.maximum(np.abs(delta), 1e-12))))
        # Preserve the planned segment direction. Independent clipping cuts corners.
        return ActionChunk((tuple(delta * scale),), self.control_ticks)

    def summary(self):
        return dict(
            name=self.plan.name,
            status=self.status,
            reason=self.reason,
            stage_index=self.stage_index,
            stage_count=len(self.plan.stages),
            stage_name=self.stage.name if self.status != "completed" else "Task complete",
            owner=self.stage.owner if self.status != "completed" else None,
            waypoint=self.waypoint,
            waypoint_count=len(self.stage.path),
            minimum_clearance_m=self.minimum_clearance,
            stages=[
                dict(name=s.name, owner=s.owner, method=s.method, waypoints=len(s.path))
                for s in self.plan.stages
            ],
            events=list(self.events),
            kind=self.plan.kind,
            station=list(self.plan.station),
        )
