"""The original reference motion expressed as ordinary environment actions."""

import numpy as np

from .task import TABLE_Z, TARGET


class ReferenceController:
    labels = (
        "Approach",
        "Align fingers",
        "Close gripper",
        "Lift",
        "Carry",
        "Lower",
        "Release",
        "Withdraw",
        "Verify placement",
    )

    def __init__(self, env):
        pick = env.data.xpos[env.block].copy()
        self.phases = [
            (1.8, pick + [0, 0, 0.17], 0.08),
            (1.4, pick + [0, 0, 0.003], 0.08),
            (0.9, pick + [0, 0, 0.003], 0),
            (1.6, pick + [0, 0, 0.20], 0),
            (2.0, TARGET + [0, 0, 0.20], 0),
            (1.5, TARGET + [0, 0, 0.003], 0),
            (0.9, TARGET + [0, 0, 0.003], 0.08),
            (1.5, TARGET + [0, 0, 0.22], 0.08),
            (0.6, TARGET + [0, 0, 0.22], 0.08),
        ]
        self.phase, self.elapsed, self.grasp_seconds = 0, 0.0, 0.0
        self.begin(env)

    def begin(self, env):
        self.start = env.data.qpos[:7].copy()
        self.goal = env.solve(self.phases[self.phase][1], self.start)
        self.elapsed = 0.0

    def action(self, env):
        duration, _, gap = self.phases[self.phase]
        t = min(1.0, (self.elapsed + env.dt) / duration)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        return np.r_[self.start + blend * (self.goal - self.start), gap]

    def advance(self, env):
        self.elapsed += env.dt
        if self.phase == 2 and len(env.contacts()) == 2:
            self.grasp_seconds += env.dt
        if self.elapsed + 1e-9 < self.phases[self.phase][0]:
            return
        if self.phase == 2 and self.grasp_seconds < 0.1:
            raise ValueError("Grasp missed: both fingers did not establish contact")
        if self.phase in (3, 4) and env.data.xpos[env.block, 2] < TABLE_Z + 0.12:
            raise ValueError("The block slipped before placement")
        if self.phase == len(self.phases) - 1:
            raise ValueError("Placement did not settle inside the target")
        self.phase += 1
        self.begin(env)
