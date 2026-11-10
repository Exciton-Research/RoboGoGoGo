"""A shortcut timeout must preserve only previously certified tree edges."""

from types import SimpleNamespace

import numpy as np
import pytest

from robogogogo.planning.motion import PlanningError, plan_motion


class BudgetWorld:
    lower = np.array([0.0])
    upper = np.array([1.0])
    margin = 0.01

    def __init__(self, message):
        self.message, self.checks, self.edges = message, 0, set()

    def clearance(self, q):
        return SimpleNamespace(distance=1.0, pair=("clear", "clear"))

    def edge_free(self, a, b):
        self.checks += 1
        if self.checks == 1:
            return False
        if np.linalg.norm(b - a) > 0.251:
            raise PlanningError(self.message)
        self.edges.add(tuple(sorted((float(a[0]), float(b[0])))))
        return True


def test_shortcut_budget_exhaustion_returns_certified_tree_path():
    world = BudgetWorld("Collision edge certification exceeded the planning budget")
    plan = plan_motion(world, [0.0], [1.0], step_size=0.25)
    assert len(plan.waypoints) > 2
    assert plan.waypoints[0] == (0.0,) and plan.waypoints[-1] == (1.0,)
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        assert tuple(sorted((a[0], b[0]))) in world.edges


def test_shortcut_does_not_hide_other_errors():
    with pytest.raises(PlanningError, match="invalid geometry"):
        plan_motion(BudgetWorld("invalid geometry"), [0.0], [1.0], step_size=0.25)
