"""The original reference motion expressed as ordinary environment actions."""

import numpy as np

from .task import HOME


class PickupFailure(ValueError):
    """Measured finger opening no longer supports a stable grasp."""


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

    def __init__(self, env, *, camera_pick=None, camera_grasp=None):
        self.camera_based = camera_pick is not None
        self.perception = "camera" if self.camera_based else "simulator"
        pick = (
            np.asarray(camera_pick, dtype=float).copy()
            if self.camera_based
            else env.data.xpos[env.block].copy()
        )
        self.pick = pick.copy()
        grasp = pick if camera_grasp is None else np.asarray(camera_grasp, dtype=float)
        self.grasp_offset = np.zeros(3)
        self.compliant_hold = False
        self.slip_seconds = 0.0
        self.repositioning = self.camera_based and np.linalg.norm(env.data.qpos[:7] - HOME) > 0.3
        self.labels = list(type(self).labels)
        if self.repositioning:
            self.labels[0] = "Return above table"
        target = env.target
        self.upper_grasp = self.camera_based and grasp[2] > pick[2] + 0.01
        lift, carry = grasp + [0, 0, 0.20], target + [0, 0, 0.20]
        if self.upper_grasp:
            # Keep the object above the supported 22 cm tabletop clutter while
            # translating. Upper grasps need their tool-to-center offset too;
            # descending toward the destination during carry can strike a carton.
            travel_z = max(grasp[2], target[2] + max(0, grasp[2] - pick[2])) + 0.25
            lift[2] = carry[2] = travel_z
        self.phases = [
            (1.8, grasp + [0, 0, 0.17], 0.08),
            (1.4, grasp + [0, 0, 0.003], 0.08),
            (0.9, grasp + [0, 0, 0.003], 0),
            (1.6, lift, 0),
            (2.0, carry, 0),
            (1.5, target + [0, 0, 0.003], 0),
            (0.9, target + [0, 0, 0.003], 0.08),
            (1.5, target + [0, 0, 0.22], 0.08),
            (6.0, target + [0, 0, 0.22], 0.08),
        ]
        self.phase, self.elapsed, self.grasp_seconds = 0, 0.0, 0.0
        self.begin(env)

    def begin(self, env):
        self.start = env.data.qpos[:7].copy()
        self.start_position = env.data.site_xpos[env.site].copy()
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
        self.goal = (
            HOME.copy()
            if self.phase == 0 and self.repositioning
            else env.solve(self.phases[self.phase][1], self.start)
        )
        self.elapsed = 0.0

    def action(self, env):
        duration, _, gap = self.phases[self.phase]
        t = min(1.0, (self.elapsed + env.dt) / duration)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        if (
            self.camera_based
            and not self.upper_grasp
            and gap == 0
            and (self.phase == 2 or self.compliant_hold)
        ):
            # Lower grasps on small/rounded objects need less squeeze. A target
            # halfway toward the measured opening halves the closing preload,
            # using finger proprioception rather than catalog dimensions.
            gap = float(np.clip(sum(env.data.qpos[7:9]) * 0.5, 0, 0.08))
        joints = self.start + blend * (self.goal - self.start)
        if self.camera_based and not self.repositioning and self.phase in (0, 1, 3, 4, 5, 7):
            # Joint interpolation can tilt the hand between two upright IK
            # endpoints and roll a rounded object out. Track the Cartesian line
            # with the fixed grasp orientation using robot proprioception only.
            position = self.start_position + blend * (
                self.phases[self.phase][1] - self.start_position
            )
            joints = env.solve(position, env.data.qpos[:7])
        return np.r_[joints, gap]

    def advance(self, env):
        self.elapsed += env.dt
        if self.camera_based and self.phase in (3, 4, 5):
            self.slip_seconds = (
                self.slip_seconds + env.dt if sum(env.data.qpos[7:9]) < 0.008 else 0.0
            )
            if self.slip_seconds >= 0.15:
                raise PickupFailure("Gripper closed empty during transport")
        grasped = (
            0.008 < sum(env.data.qpos[7:9]) < 0.078
            if self.camera_based
            else len(env.contacts()) == 2
        )
        if self.phase == 2 and grasped:
            self.grasp_seconds += env.dt
        if self.elapsed + 1e-9 < self.phases[self.phase][0]:
            return
        if self.phase == 0 and self.repositioning:
            self.repositioning = False
            self.labels[0] = "Approach"
            self.begin(env)
            return
        if self.phase == 2 and self.grasp_seconds < 0.1:
            raise PickupFailure("Grasp missed: gripper did not establish a stable hold")
        if self.phase == 2 and self.camera_based:
            # Narrow rounded objects can be squeezed out by excess preload.
            # Finger opening is measured after closure; no object dimensions.
            self.compliant_hold = not self.upper_grasp and sum(env.data.qpos[7:9]) < 0.05
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
