"""Seeded bidirectional RRT-Connect in the combined configuration of every arm."""

import time
from dataclasses import dataclass

import numpy as np


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class MotionPlan:
    waypoints: tuple[tuple[float, ...], ...]
    method: str
    checks: int


def plan_motion(world, start, goal, *, seed=0, max_iterations=1500, step_size=0.25):
    """Bounded search; every returned edge is checked, including shortcuts.

    A search failure is not proof that no path exists. Seed and geometry determine
    reproducibility. The planner does not impose end-effector orientation constraints.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if max_iterations < 1 or not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("Invalid planning budget or step size")
    deadline = time.monotonic() + 15.0
    initial = world.checks
    for label, q in [("start", start), ("goal", goal)]:
        clearance = world.clearance(q)
        if clearance.distance < world.margin:
            raise PlanningError(
                f"{label} violates clearance: {clearance.pair} ({clearance.distance:.4f} m)"
            )
    if world.edge_free(start, goal):
        return MotionPlan((tuple(start), tuple(goal)), "direct", world.checks - initial)
    rng = np.random.default_rng(seed)
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        distance = np.linalg.norm(delta)
        if distance < 1e-10:
            return nearest, True
        following = nodes[nearest] + delta * min(1.0, step_size / distance)
        if not world.edge_free(nodes[nearest], following):
            return None, False
        nodes.append(following)
        parents.append(nearest)
        return len(nodes) - 1, distance <= step_size

    def branch(tree, index):
        path = []
        while index >= 0:
            path.append(tree[0][index])
            index = tree[1][index]
        return path[::-1]

    for iteration in range(max_iterations):
        if time.monotonic() > deadline:
            raise PlanningError("Motion planning exceeded its 15 second budget")
        side = iteration % 2
        sample = (
            trees[1 - side][0][0] if iteration % 5 == 0 else rng.uniform(world.lower, world.upper)
        )
        index, _ = extend(trees[side], sample)
        if index is None:
            continue
        target = trees[side][0][index]
        for _ in range(100):
            other, reached = extend(trees[1 - side], target)
            if other is None:
                break
            if reached:
                paths = [None, None]
                paths[side] = branch(trees[side], index)
                paths[1 - side] = branch(trees[1 - side], other)
                path = paths[0] + paths[1][-2::-1]
                # Deterministic greedy shortcutting retains certified edges.
                shortened = [path[0]]
                cursor = 0
                try:
                    while cursor < len(path) - 1:
                        endpoint = len(path) - 1
                        while endpoint > cursor + 1 and not world.edge_free(
                            path[cursor], path[endpoint]
                        ):
                            endpoint -= 1
                        shortened.append(path[endpoint])
                        cursor = endpoint
                except (PlanningError, TimeoutError) as error:
                    if "budget" not in str(error).lower():
                        raise
                    # Every raw tree edge was already certified before insertion.
                    # Optional shortcutting must not discard a valid found path
                    # when the tighter internal geometry exhausts its time budget.
                    shortened = path
                return MotionPlan(
                    tuple(tuple(q) for q in shortened), "rrt_connect", world.checks - initial
                )
    raise PlanningError(f"No collision-free path found within {max_iterations} iterations")
