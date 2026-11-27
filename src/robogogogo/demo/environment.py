"""Explicitly stepped Panda environment. No clocks, browser, or recording side effects."""

import base64
import hashlib
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .assets import SOURCE, require_model
from .task import HALF_CUBE, HOME, TABLE_Z, TARGET, scene_xml

CAMERAS = ("overhead", "wrist")
WIDTH, HEIGHT = 240, 144
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class PandaEnv(gym.Env):
    """Seven absolute arm targets (rad) and total finger opening (m), held for 50 ms.

    Agent observations contain robot proprioception, previous commands, and
    optional RGB. Object state and task diagnostics live in info, not observations.
    One instance belongs to one thread; independent instances can run independently.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, assets=None, *, observation_mode="state", render_mode=None, max_steps=800):
        if observation_mode not in ("state", "rgb") or render_mode not in (None, "rgb_array"):
            raise ValueError(
                "observation_mode must be state/rgb; render_mode must be rgb_array/None"
            )
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        asset_path = require_model(assets or Path.cwd() / "runs/demo/assets")
        xml = scene_xml(asset_path)
        self.model_hash = hashlib.sha256(
            xml.replace(str(asset_path.parent / "assets"), "@panda/assets").encode()
        ).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.site = self.model.site("grasp_center").id
        self.block, self.hand = self.model.body("block").id, self.model.body("hand").id
        self.fingers = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.visual_ids = [i for i in range(self.model.ngeom) if self.model.geom_group[i] != 3]
        self.control_steps, self.dt = 25, float(self.model.opt.timestep) * 25
        self.max_steps, self.observation_mode, self.render_mode = (
            max_steps,
            observation_mode,
            render_mode,
        )
        self.renderer = None
        self._geometry = None
        self.action_space = spaces.Box(
            np.r_[self.model.jnt_range[:7, 0], 0.0],
            np.r_[self.model.jnt_range[:7, 1], 0.08],
            dtype=np.float64,
        )
        fields = {
            "proprio": spaces.Box(-np.inf, np.inf, (25,), np.float64),
            "previous_action": self.action_space,
            "time": spaces.Box(0.0, np.inf, (1,), np.float64),
        }
        if observation_mode == "rgb":
            fields.update(
                {name: spaces.Box(0, 255, (HEIGHT, WIDTH, 3), np.uint8) for name in CAMERAS}
            )
        self.observation_space = spaces.Dict(fields)
        self.reset(seed=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"block_xy"}:
            raise ValueError("Only block_xy is supported in reset options")
        xy = np.asarray(
            options.get(
                "block_xy",
                np.array([0.47, -0.17])
                + (self.np_random.uniform(-0.035, 0.035, 2) if seed != 0 else 0),
            ),
            dtype=float,
        )
        if (
            xy.shape != (2,)
            or not np.all(np.isfinite(xy))
            or not (0.40 <= xy[0] <= 0.54 and -0.24 <= xy[1] <= -0.10)
        ):
            raise ValueError("block_xy must be within x=[0.40,0.54], y=[-0.24,-0.10] m")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7], self.data.qpos[7:9] = HOME, 0.04
        self.data.qpos[9:12], self.data.qpos[12:16] = (
            [*xy, TABLE_Z + HALF_CUBE + 0.002],
            [1, 0, 0, 0],
        )
        self.data.ctrl[:7], self.data.ctrl[7] = HOME, 255
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.orientation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        self.pick = self.data.xpos[self.block].copy()
        self.max_height, self.settled_steps, self.steps = float(self.pick[2]), 0, 0
        self.previous_action = np.r_[HOME, 0.08]
        self.terminated = self.truncated = False
        return self.observe(), self.info()

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended; reset before stepping again")
        action = np.asarray(action, dtype=np.float64)
        if (
            action.shape != (8,)
            or not np.all(np.isfinite(action))
            or not self.action_space.contains(action)
        ):
            raise ValueError(
                "Action requires seven joint targets in rad and finger opening in [0,0.08] m"
            )
        self.data.ctrl[:7], self.data.ctrl[7] = action[:7], action[7] / 0.08 * 255
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)
            self.max_height = max(self.max_height, float(self.data.xpos[self.block, 2]))
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.previous_action = action.copy()
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("Nonfinite simulator state; reset required")
        placed = self._placed()
        self.settled_steps = self.settled_steps + 1 if placed else 0
        fallen = bool(self.data.xpos[self.block, 2] < TABLE_Z - 0.08)
        self.terminated = self.settled_steps >= 6 or fallen
        self.truncated = self.steps >= self.max_steps and not self.terminated
        info = self.info()
        return self.observe(), float(info["success"]), self.terminated, self.truncated, info

    def _placed(self):
        position = self.data.xpos[self.block]
        return bool(
            np.linalg.norm(position[:2] - TARGET[:2]) < 0.035
            and abs(position[2] - TARGET[2]) < 0.008
            and np.linalg.norm(self.data.qvel[9:12]) < 0.02
            and sum(self.data.qpos[7:9]) > 0.065
            and not self.contacts()
            and self.max_height > TABLE_Z + 0.12
            and np.linalg.norm(self.data.site_xpos[self.site] - position) > 0.12
        )

    def contacts(self):
        touching = set()
        for contact in self.data.contact:
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self.block in bodies:
                touching.update(bodies & self.fingers)
        return touching

    def info(self):
        return {
            "step_index": self.steps,
            "time": self.steps * self.dt,
            "success": self.settled_steps >= 6,
            "reason": "placed"
            if self.settled_steps >= 6
            else "fallen"
            if self.terminated
            else "time_limit"
            if self.truncated
            else None,
            "block": self.data.xpos[self.block].copy(),
            "target": TARGET.copy(),
        }

    def observe(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        observation = {
            "proprio": np.r_[
                self.data.qpos[:9],
                self.data.qvel[:9],
                self.data.site_xpos[self.site],
                quat[[1, 2, 3, 0]],
            ],
            "previous_action": self.previous_action.copy(),
            "time": np.array([self.steps * self.dt]),
        }
        if self.observation_mode == "rgb":
            observation.update(self.render_cameras())
        return observation

    def render(self):
        return self.render_cameras()["overhead"]

    def render_cameras(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
            self.render_options = mujoco.MjvOption()
            self.render_options.geomgroup[3] = 0
        frames = {}
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=name, scene_option=self.render_options)
            frames[name] = self.renderer.render().copy()
        return frames

    def capture_state(self):
        state = np.zeros(mujoco.mj_stateSize(self.model, STATE_SPEC))
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        return {
            "integration": state,
            "aux": np.array([self.steps, self.max_height, self.settled_steps]),
            "pick": self.pick.copy(),
            "orientation": self.orientation.copy(),
            "previous_action": self.previous_action.copy(),
            "ended": np.array([self.terminated, self.truncated], dtype=bool),
        }

    def restore_state(self, state):
        integration = np.asarray(state["integration"], dtype=np.float64)
        if integration.shape != (mujoco.mj_stateSize(self.model, STATE_SPEC),) or not np.all(
            np.isfinite(integration)
        ):
            raise ValueError("Invalid integration state")
        mujoco.mj_setState(self.model, self.data, integration, STATE_SPEC)
        self.steps, self.max_height, self.settled_steps = state["aux"]
        self.steps, self.settled_steps = int(self.steps), int(self.settled_steps)
        self.pick, self.orientation = state["pick"].copy(), state["orientation"].copy()
        self.previous_action = state["previous_action"].copy()
        self.terminated, self.truncated = map(bool, state["ended"])
        mujoco.mj_forward(self.model, self.data)

    def solve(self, position, initial):
        """IK helper for manual and reference controllers; not called by step()."""
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("A finite Cartesian target is required")
        scratch, model = self.scratch, self.model
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[:7] = initial
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(160):
            mujoco.mj_forward(model, scratch)
            error_rotation = self.orientation @ scratch.site_xmat[self.site].reshape(3, 3).T
            angular = 0.5 * np.array(
                [
                    error_rotation[2, 1] - error_rotation[1, 2],
                    error_rotation[0, 2] - error_rotation[2, 0],
                    error_rotation[1, 0] - error_rotation[0, 1],
                ]
            )
            linear = position - scratch.site_xpos[self.site]
            if np.linalg.norm(linear) < 0.0003 and np.linalg.norm(angular) < 0.002:
                return scratch.qpos[:7].copy()
            mujoco.mj_jacSite(model, scratch, jp, jr, self.site)
            jac = np.vstack((jp[:, :7], 0.5 * jr[:, :7]))
            update = jac.T @ np.linalg.solve(
                jac @ jac.T + np.eye(6) * 1e-5, np.r_[linear, 0.5 * angular]
            )
            scratch.qpos[:7] = np.clip(
                scratch.qpos[:7] + np.clip(update, -0.10, 0.10),
                model.jnt_range[:7, 0] + 0.01,
                model.jnt_range[:7, 1] - 0.01,
            )
        raise ValueError("Target is outside the controller's reachable workspace")

    def snapshot(self):
        poses, cameras, quat = [], {}, np.zeros(4)
        for geom in self.visual_ids:
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[geom])
            poses.append([*self.data.geom_xpos[geom].tolist(), *quat[[1, 2, 3, 0]].tolist()])
        for name in CAMERAS:
            camera = self.model.camera(name).id
            mujoco.mju_mat2Quat(quat, self.data.cam_xmat[camera])
            cameras[name] = {
                "position": self.data.cam_xpos[camera].tolist(),
                "quaternion": quat[[1, 2, 3, 0]].tolist(),
                "fovy": float(self.model.cam_fovy[camera]),
            }
        return {
            "poses": poses,
            "cameras": cameras,
            "time": self.steps * self.dt,
            "step_index": self.steps,
            "block": self.data.xpos[self.block].tolist(),
            "tool": self.data.site_xpos[self.site].tolist(),
            "hand_position": self.data.xpos[self.hand].tolist(),
            "hand_rotation": self.data.xmat[self.hand].tolist(),
            "target": TARGET.tolist(),
            "pick": self.pick.tolist(),
            "finger_gap_mm": float(sum(self.data.qpos[7:9]) * 1000),
            "lift_mm": max(0, float(self.data.xpos[self.block, 2] - self.pick[2]) * 1000),
            "placement_error_mm": float(
                np.linalg.norm(self.data.xpos[self.block, :2] - TARGET[:2]) * 1000
            ),
            "finger_contacts": len(self.contacts()),
            "joint_target": self.previous_action.tolist(),
        }

    def geometry(self):
        if self._geometry is not None:
            return self._geometry
        model, meshes, geoms = self.model, {}, []
        for geom in self.visual_ids:
            mesh, kind = int(model.geom_dataid[geom]), int(model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH) and str(mesh) not in meshes:
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                face, nface = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                meshes[str(mesh)] = {
                    "vertices": base64.b64encode(
                        np.asarray(model.mesh_vert[start : start + count], dtype="<f4").tobytes()
                    ).decode(),
                    "faces": base64.b64encode(
                        np.asarray(model.mesh_face[face : face + nface], dtype="<u4").tobytes()
                    ).decode(),
                }
            material = int(model.geom_matid[geom])
            geoms.append(
                {
                    "type": kind,
                    "mesh": mesh,
                    "size": model.geom_size[geom].tolist(),
                    "rgba": (
                        model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
                    ).tolist(),
                    "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                }
            )
        self._geometry = {"meshes": meshes, "geoms": geoms, "source": SOURCE}
        return self._geometry

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
