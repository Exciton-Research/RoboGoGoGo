"""Panda Lift adapter for robosuite 1.5.1; imported only by an explicit run.

Nominal MuJoCo parameters are a reference condition, not measured ground truth.
The scripted demonstrator sees cube position. Learned policies never do.
"""

import hashlib
import json
from collections import deque
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

INSTRUCTION = "Grasp the cube, lift it above the table, and keep holding it."
PROPRIO_KEYS = (
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_eef_pos",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
)


class LiftExpert:
    """Unvalidated state-based grasp script; all successes/failures are retained."""

    def __init__(self, initial):
        self.origin = np.asarray(initial["cube_pos"]).copy()
        self.phase, self.closing_steps = 0, 0

    def act(self, raw):
        cube, eef = np.asarray(raw["cube_pos"]), np.asarray(raw["robot0_eef_pos"])
        goal = cube + [0, 0, 0.12] if self.phase == 0 else cube + [0, 0, 0.005]
        if self.phase == 0 and np.linalg.norm(goal - eef) < 0.015:
            self.phase = 1
            goal = cube + [0, 0, 0.005]
        if self.phase == 1 and np.linalg.norm(goal - eef) < 0.012:
            self.phase = 2
        if self.phase == 2:
            self.closing_steps += 1
            if self.closing_steps >= 12:
                self.phase = 3
        if self.phase == 3:
            goal = self.origin + [0, 0, 0.20]
        action = np.zeros(7)
        action[:3] = np.clip((goal - eef) / 0.05, -1, 1)
        action[-1] = 1.0 if self.phase >= 2 else -1.0
        return action


class RobosuiteBackend:
    def __init__(self, study):
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        if version("robosuite") != "1.5.1":
            raise RuntimeError("This adapter targets robosuite==1.5.1; use the documented version")
        self.study = study
        self.make = robosuite.make
        self.controller = load_composite_controller_config(robot="Panda")
        # The expert's Cartesian error and actions are expressed in world axes.
        self.controller["body_parts"]["right"]["input_ref_frame"] = "world"

    def metadata(self):
        return {
            "backend": "robosuite",
            "task": "Lift",
            "robot": "Panda",
            "versions": {key: version(key) for key in ("robosuite", "mujoco", "numpy")},
            "camera": "agentview",
            "proprio_keys": list(PROPRIO_KEYS),
            "proprio_components": [
                {"key": PROPRIO_KEYS[0], "dimensions": 7, "units": "unitless"},
                {"key": PROPRIO_KEYS[1], "dimensions": 7, "units": "unitless"},
                {"key": PROPRIO_KEYS[2], "dimensions": 3, "units": "m", "frame": "world"},
                {"key": PROPRIO_KEYS[3], "dimensions": 4, "ordering": "xyzw", "frame": "world"},
                {"key": PROPRIO_KEYS[4], "dimensions": 2, "units": "m"},
            ],
            "instruction": INSTRUCTION,
            "action": {
                "controller": self.controller,
                "dimensions": 7,
                "meaning": "normalized OSC_POSE xyz/rotation deltas, then gripper command",
                "requested_vs_executed": "executed commands pass through the delay queue",
                "max_xyz_delta_m": 0.05,
            },
            "demonstrator": "privileged state-based heuristic, not a learned policy",
            "policy_inputs": ["rgb", "proprio", "previous_action", "instruction"],
            "success": "sparse Lift success at every step of the final hold window",
            "scope": "sim-to-sim only; no real-robot or VLA result is implied",
        }

    def _observe(self, raw, previous):
        return {
            "rgb": np.asarray(raw["agentview_image"], dtype=np.uint8).copy(),
            "proprio": np.concatenate([np.asarray(raw[key]).ravel() for key in PROPRIO_KEYS]),
            "previous_action": np.asarray(previous).copy(),
        }

    def rollout(self, profile, seed, *, policy=None, save=None):
        """Run exactly horizon commands; optional archive stores H+1 observations."""
        study, started = self.study, perf_counter()
        env = None
        # Some upstream components also use legacy numpy randomness. Restore the
        # caller's stream, and seed before construction as well as env.reset().
        random_state = np.random.get_state()
        np.random.seed(seed)
        try:
            env = self.make(
                "Lift",
                robots="Panda",
                controller_configs=deepcopy(self.controller),
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                use_object_obs=True,
                camera_names="agentview",
                camera_heights=study.image_size,
                camera_widths=study.image_size,
                control_freq=study.control_hz,
                horizon=study.horizon,
                reward_shaping=False,
                hard_reset=False,
                initialization_noise=None,
                seed=seed,
            )
            if env.action_dim != 7:
                raise RuntimeError("Panda OSC_POSE adapter requires seven command dimensions")
            raw = env.reset()
            model = env.sim.model
            model.geom_friction[:, 0] *= profile["contact_friction_scale"]
            camera = model.camera_name2id("agentview")
            model.cam_pos[camera, 0] += profile["camera_x_offset_m"]
            model.light_diffuse[:] *= profile["light_scale"]
            model.light_ambient[:] *= profile["light_scale"]
            env.sim.forward()
            # reset() captured images before the interventions. Refresh sensors
            # so the very first policy observation has the same treatment.
            raw = env._get_observations(force_update=True)
            initial_qpos = env.sim.data.qpos.copy()
            initial_qvel = env.sim.data.qvel.copy()
            expert = LiftExpert(raw) if policy is None else None
            if policy is not None:
                policy.reset()
            delay = deque(np.zeros(7) for _ in range(profile["action_delay_steps"]))
            observation = self._observe(raw, np.zeros(7))
            observations = [observation] if save is not None else None
            requested, executed, rewards, successes = [], [], [], []
            for _ in range(study.horizon):
                if expert is not None:
                    action = expert.act(raw)
                else:
                    # No raw simulator state, object position, reward, or success
                    # is passed to the learned policy.
                    action = policy.act(observation, INSTRUCTION)
                action = np.asarray(action, dtype=float)
                if action.shape != (7,) or not np.all(np.isfinite(action)):
                    raise ValueError("Policy must return seven finite OSC_POSE commands")
                low, high = env.action_spec
                if np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
                    raise ValueError("Policy command outside action_spec; no silent clipping")
                delay.append(action.copy())
                applied = delay.popleft()
                raw, reward, done, _ = env.step(applied)
                observation = self._observe(raw, action)
                requested.append(action.copy())
                executed.append(applied.copy())
                rewards.append(float(reward))
                successes.append(bool(reward > 0))
                if observations is not None:
                    observations.append(observation)
                if done and len(rewards) != study.horizon:
                    raise RuntimeError("Unexpected early termination; fixed-budget trial aborted")
            success = all(successes[-study.success_hold_steps :])
            result = {
                "seed": seed,
                "profile": dict(profile),
                "success": success,
                "initial_state_sha256": hashlib.sha256(
                    initial_qpos.tobytes() + initial_qvel.tobytes()
                ).hexdigest(),
                "ever_success": any(successes),
                "failure": None
                if success
                else ("lift_not_held" if any(successes) else "never_lifted"),
                "return": float(sum(rewards)),
                "steps": len(rewards),
                "wall_seconds": perf_counter() - started,
            }
            if save is not None:
                path = Path(save)
                np.savez_compressed(
                    path,
                    **{key: np.stack([obs[key] for obs in observations]) for key in observation},
                    requested_action=np.asarray(requested),
                    executed_action=np.asarray(executed),
                    reward=np.asarray(rewards),
                    success=np.asarray(successes),
                    initial_qpos=initial_qpos,
                    initial_qvel=initial_qvel,
                    metadata=np.asarray(json.dumps({**result, "instruction": INSTRUCTION})),
                )
            return result
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                np.random.set_state(random_state)
