"""The original reference motion expressed as ordinary environment actions."""

import numpy as np


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

    def __init__(self, env, *, camera_pick=None):
        self.camera_based = camera_pick is not None
        pick = (
            np.asarray(camera_pick, dtype=float).copy()
            if self.camera_based
            else env.data.xpos[env.block].copy()
        )
        self.pick = pick.copy()
        self.grasp_offset = np.zeros(3)
        target = env.target
        self.phases = [
            (1.8, pick + [0, 0, 0.17], 0.08),
            (1.4, pick + [0, 0, 0.003], 0.08),
            (0.9, pick + [0, 0, 0.003], 0),
            (1.6, pick + [0, 0, 0.20], 0),
            (2.0, target + [0, 0, 0.20], 0),
            (1.5, target + [0, 0, 0.003], 0),
            (0.9, target + [0, 0, 0.003], 0.08),
            (1.5, target + [0, 0, 0.22], 0.08),
            (0.6, target + [0, 0, 0.22], 0.08),
        ]
        self.phase, self.elapsed, self.grasp_seconds = 0, 0.0, 0.0
        self.begin(env)

    def begin(self, env):
        self.start = env.data.qpos[:7].copy()
        if self.phase == 2 and not self.camera_based:
            duration, _, opening = self.phases[2]
            self.phases[2] = (duration, env.data.xpos[env.block] + [0, 0, 0.003], opening)
        if self.phase == 5:
            # Lower the object's measured center, accounting for grasp slip.
            offset = (
                self.grasp_offset
                if self.camera_based
                else env.data.xpos[env.block] - env.data.site_xpos[env.site]
            )
            place_tool = env.target + [0, 0, 0.003] - offset
            for index in (5, 6):
                duration, _, opening = self.phases[index]
                self.phases[index] = (duration, place_tool.copy(), opening)
        self.goal = env.solve(self.phases[self.phase][1], self.start)
        self.elapsed = 0.0

    def action(self, env):
        duration, _, gap = self.phases[self.phase]
        t = min(1.0, (self.elapsed + env.dt) / duration)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        return np.r_[self.start + blend * (self.goal - self.start), gap]

    def advance(self, env):
        self.elapsed += env.dt
        grasped = (
            0.008 < sum(env.data.qpos[7:9]) < 0.078
            if self.camera_based
            else len(env.contacts()) == 2
        )
        if self.phase == 2 and grasped:
            self.grasp_seconds += env.dt
        if self.elapsed + 1e-9 < self.phases[self.phase][0]:
            return
        if self.phase == 2 and self.grasp_seconds < 0.1:
            raise ValueError("Grasp missed: gripper did not establish a stable hold")
        if self.phase == 2 and self.camera_based:
            # Estimate rigid grasp offset from camera pickup and tool proprioception.
            self.grasp_offset = self.pick - env.data.site_xpos[env.site]
        if (
            not self.camera_based
            and self.phase in (3, 4)
            and env.data.xpos[env.block, 2] < env.pick[2] + 0.095
        ):
            raise ValueError("The object slipped before placement")
        if self.phase == len(self.phases) - 1:
            raise ValueError("Placement did not settle inside the target")
        self.phase += 1
        self.begin(env)
