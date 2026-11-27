"""Policy boundary shared by a future VLA and the explicitly stepped environment."""

from typing import Mapping, Protocol

import numpy as np


class PandaPolicy(Protocol):
    def act(self, observation: Mapping[str, np.ndarray], *, instruction: str) -> np.ndarray:
        """Return seven absolute joint targets (rad) and total finger opening (m).

        Observations contain proprioception, prior action, time, and optional
        overhead/wrist RGB. Scene object lists and poses are not observation fields.
        Stateful policies should be reset by their owner at episode boundaries.
        """
        ...


def step_policy(env, policy: PandaPolicy):
    """One synchronous policy action, validated and executed by env.step()."""
    return env.step(policy.act(env.observe(), instruction=env.instruction))
