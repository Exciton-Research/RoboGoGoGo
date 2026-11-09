"""Task and collision-aware motion planning for fixed-base robot systems."""

from .collision import (
    BoxObstacle,
    Capsule,
    CapsuleObstacle,
    Clearance,
    CollisionWorld,
    SphereObstacle,
    adjacent_exclusions,
    capsules_from_visuals,
)
from .motion import MotionPlan, PlanningError, plan_motion

__all__ = [
    "BoxObstacle",
    "Capsule",
    "CapsuleObstacle",
    "Clearance",
    "CollisionWorld",
    "SphereObstacle",
    "adjacent_exclusions",
    "capsules_from_visuals",
    "MotionPlan",
    "PlanningError",
    "plan_motion",
]

from .task import TaskExecutor, TaskPlan, TaskStage, plan_shared_inspection

__all__ += ["TaskExecutor", "TaskPlan", "TaskStage", "plan_shared_inspection"]
