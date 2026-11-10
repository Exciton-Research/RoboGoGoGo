import numpy as np
import pytest

from robogogogo.contracts import ActionChunk, Observation
from robogogogo.policies import JointWaypointPolicy
from robogogogo.system import ControlSystem


def obs(q=(0.0,), qd=(0.0,)):
    return Observation("", b"", q, qd, 0.0)


class Proposal:
    def __init__(self, actions):
        self.actions = actions

    def predict(self, observation):
        return ActionChunk(self.actions)


def test_supervisor_bounds_every_action_in_chunk_and_stops():
    system = ControlSystem(
        {"learned": Proposal(((9.0,), (9.0,), (9.0,)))}, lower=[-1], upper=[0.25], max_delta=[0.1]
    )
    result = system.step(obs(), controller="learned")
    np.testing.assert_allclose(result.action.actions, [[0.1], [0.1], [0.05]])
    assert result.limited
    system.stop()
    with pytest.raises(ValueError, match="stopped"):
        system.step(obs(), controller="learned")
    system.resume()
    assert system.step(obs(), controller="learned").controller == "learned"


@pytest.mark.parametrize("actions", [((float("nan"),),), ((1.0, 2.0),), ()])
def test_supervisor_rejects_invalid_proposals(actions):
    system = ControlSystem({"bad": Proposal(actions)}, lower=[-1], upper=[1], max_delta=[0.1])
    with pytest.raises(ValueError):
        system.step(obs(), controller="bad")


def test_waypoints_wait_for_settling_and_hold_when_done():
    policy = JointWaypointPolicy([[0.2], [0.4]], max_delta=[0.1])
    assert policy.predict(obs()).actions == ((0.1,),)
    policy.predict(obs((0.2,), (0.5,)))
    assert policy.index == 0
    policy.predict(obs((0.2,)))
    assert policy.index == 1
    assert policy.predict(obs((0.4,))).actions == ((0.0,),)
    assert policy.done
    policy.reset()
    assert not policy.done


def test_continuous_joint_bounds_keep_finite_increment_limits():
    system = ControlSystem(
        {"wheel": Proposal(((9.0,),))}, lower=[-np.inf], upper=[np.inf], max_delta=[0.1]
    )
    result = system.step(obs((100.0,)), controller="wheel")
    np.testing.assert_allclose(result.action.actions, [[0.1]])
    for lower, upper in (([np.nan], [1]), ([np.inf], [np.inf]), ([-np.inf], [-np.inf])):
        with pytest.raises(ValueError, match="bounds"):
            ControlSystem({"wheel": Proposal(((0.0,),))}, lower=lower, upper=upper, max_delta=[0.1])
