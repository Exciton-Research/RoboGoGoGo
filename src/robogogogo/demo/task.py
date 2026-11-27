"""MuJoCo contact dynamics and a fixed, scripted pick-and-place controller.

The cube has a free joint throughout the task. It is never parented, welded,
teleported, or attached to the hand. Only reset writes its position.
"""

import base64
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .assets import SOURCE, require_model

TABLE_Z = 0.74
HALF_CUBE = 0.025
TARGET = np.array([0.47, 0.19, TABLE_Z + HALF_CUBE])
HOME = np.array([0, -0.35, 0, -2.1, 0, 1.75, -0.7854])


def scene_xml(path):
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    root.find("option").set("iterations", "60")
    root.remove(root.find("keyframe"))
    world = root.find("worldbody")
    world.find("body").set("pos", f"0 0 {TABLE_Z}")
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_center", pos="0 0 0.1034", size="0.004")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="5 5 0.1", rgba="0.88 0.87 0.84 1"
    )
    ET.SubElement(
        world,
        "geom",
        name="worktop",
        type="box",
        size="0.66 0.48 0.035",
        pos=f"0.3 0 {TABLE_Z - 0.035}",
        rgba="0.72 0.74 0.72 1",
        friction="0.8 0.005 0.0001",
    )
    for x in (-0.25, 0.85):
        for y in (-0.37, 0.37):
            ET.SubElement(
                world,
                "geom",
                type="box",
                size="0.025 0.025 0.335",
                pos=f"{x} {y} 0.335",
                rgba="0.22 0.25 0.25 1",
            )
    cube = ET.SubElement(
        world, "body", name="block", pos=f"0.47 -0.17 {TABLE_Z + HALF_CUBE + 0.002}"
    )
    ET.SubElement(cube, "freejoint", name="block_free")
    ET.SubElement(
        cube,
        "geom",
        name="block_geom",
        type="box",
        size="0.025 0.025 0.025",
        mass="0.08",
        rgba="0.88 0.31 0.13 1",
        friction="1.1 0.01 0.0001",
        condim="4",
        solref="0.005 1",
    )
    return ET.tostring(root, encoding="unicode")


@dataclass
class Phase:
    name: str
    label: str
    seconds: float
    position: np.ndarray
    opening: float


class PickPlace:
    def __init__(self, cache):
        self.model = mujoco.MjModel.from_xml_string(scene_xml(require_model(cache)))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.site = self.model.site("grasp_center").id
        self.block = self.model.body("block").id
        self.hand = self.model.body("hand").id
        self.fingers = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.visual_ids = [i for i in range(self.model.ngeom) if self.model.geom_group[i] != 3]
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self._geometry = None
        self.reset()
        self.thread = threading.Thread(target=self._clock, name="panda-demo", daemon=True)
        self.thread.start()

    def reset(self, variant=0):
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = HOME
            self.data.qpos[7:9] = 0.04
            self.data.ctrl[:7] = HOME
            self.data.ctrl[7] = 255
            rng = np.random.default_rng(variant)
            xy = np.array([0.47, -0.17]) + (rng.uniform(-0.035, 0.035, 2) if variant else 0)
            self.data.qpos[9:12] = [*xy, TABLE_Z + HALF_CUBE + 0.002]
            self.data.qpos[12:16] = [1, 0, 0, 0]
            for _ in range(250):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.orientation = self.data.site_xmat[self.site].reshape(3, 3).copy()
            self.pick = self.data.xpos[self.block].copy()
            self.max_height = float(self.pick[2])
            self.status, self.message = "ready", "Ready to pick up the block."
            self.phase_index, self.phase_elapsed = 0, 0.0
            self.task_time, self.grasp_seconds = 0.0, 0.0
            self.start_q = self.data.qpos[:7].copy()
            self.goal_q = self.start_q.copy()
            self.phases = [
                Phase("approach", "Approach", 1.8, self.pick + [0, 0, 0.17], 255),
                Phase("descend", "Align fingers", 1.4, self.pick + [0, 0, 0.003], 255),
                Phase("grasp", "Close gripper", 0.9, self.pick + [0, 0, 0.003], 0),
                Phase("lift", "Lift", 1.6, self.pick + [0, 0, 0.20], 0),
                Phase("transfer", "Carry", 2.0, TARGET + [0, 0, 0.20], 0),
                Phase("lower", "Lower", 1.5, TARGET + [0, 0, 0.003], 0),
                Phase("release", "Release", 0.9, TARGET + [0, 0, 0.003], 255),
                Phase("retreat", "Withdraw", 1.5, TARGET + [0, 0, 0.22], 255),
                Phase("settle", "Verify placement", 0.6, TARGET + [0, 0, 0.22], 255),
            ]

    def solve(self, position, initial):
        """Damped Jacobian IK; physical actuators track its joint-space target."""
        model, scratch = self.model, self.ik_data
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[:7] = initial
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(160):
            mujoco.mj_forward(model, scratch)
            rotation = scratch.site_xmat[self.site].reshape(3, 3)
            error_rotation = self.orientation @ rotation.T
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
            error = np.r_[linear, 0.5 * angular]
            update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            scratch.qpos[:7] += np.clip(update, -0.10, 0.10)
            scratch.qpos[:7] = np.clip(
                scratch.qpos[:7], model.jnt_range[:7, 0] + 0.01, model.jnt_range[:7, 1] - 0.01
            )
        raise ValueError("This placement is outside the scripted controller's reachable workspace")

    def command(self, command, variant=0):
        with self.lock:
            if command == "reset":
                self.reset(variant)
            elif command == "pause" and self.status == "running":
                self.status = "paused"
            elif command == "run" and self.status in ("ready", "paused"):
                if self.status == "ready":
                    self._begin_phase()
                self.status = "running"
            else:
                raise ValueError("Reset before starting another task")
            return self.snapshot()

    def _begin_phase(self):
        phase = self.phases[self.phase_index]
        self.start_q = self.data.qpos[:7].copy()
        self.goal_q = self.solve(phase.position, self.start_q)
        self.phase_elapsed = 0.0
        self.message = phase.label

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

    def _step(self):
        phase = self.phases[self.phase_index]
        dt = self.model.opt.timestep
        self.phase_elapsed += dt
        self.task_time += dt
        t = min(1.0, self.phase_elapsed / phase.seconds)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        self.data.ctrl[:7] = self.start_q + blend * (self.goal_q - self.start_q)
        self.data.ctrl[7] = phase.opening
        mujoco.mj_step(self.model, self.data)
        position = self.data.xpos[self.block]
        self.max_height = max(self.max_height, float(position[2]))
        if phase.name == "grasp" and len(self.contacts()) == 2:
            self.grasp_seconds += dt
        if not np.all(np.isfinite(self.data.qpos)):
            self.status, self.message = "failed", "Simulation stopped: nonfinite state."
            return
        if self.phase_elapsed + 1e-9 < phase.seconds:
            return
        if phase.name == "grasp" and self.grasp_seconds < 0.1:
            self.status, self.message = (
                "failed",
                "Grasp missed: both fingers did not establish contact.",
            )
            return
        if phase.name in ("lift", "transfer") and position[2] < TABLE_Z + 0.12:
            self.status, self.message = "failed", "The block slipped before placement."
            return
        self.phase_index += 1
        if self.phase_index == len(self.phases):
            velocity = self.data.qvel[9:12]
            placed = (
                np.linalg.norm(position[:2] - TARGET[:2]) < 0.035
                and abs(position[2] - TARGET[2]) < 0.008
                and np.linalg.norm(velocity) < 0.02
                and sum(self.data.qpos[7:9]) > 0.065
                and not self.contacts()
                and self.max_height > TABLE_Z + 0.12
            )
            self.status = "complete" if placed else "failed"
            self.message = (
                "Block placed and released."
                if placed
                else "Placement did not settle inside the target."
            )
            self.phase_index -= 1
        else:
            try:
                self._begin_phase()
            except ValueError as error:
                self.status, self.message = "failed", str(error)

    def _clock(self):
        while not self.closed.is_set():
            started = time.perf_counter()
            with self.lock:
                if self.status == "running":
                    for _ in range(10):
                        self._step()
                        if self.status != "running":
                            break
            self.closed.wait(max(0.001, 0.02 - (time.perf_counter() - started)))

    def snapshot(self):
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            poses = []
            quat = np.zeros(4)
            for geom in self.visual_ids:
                mujoco.mju_mat2Quat(quat, self.data.geom_xmat[geom])
                poses.append([*self.data.geom_xpos[geom].tolist(), *quat[[1, 2, 3, 0]].tolist()])
            return {
                "status": self.status,
                "message": self.message,
                "phase": self.phase_index,
                "time": self.task_time,
                "phases": [phase.label for phase in self.phases],
                "progress": (
                    self.phase_index
                    + min(1, self.phase_elapsed / self.phases[self.phase_index].seconds)
                )
                / len(self.phases),
                "poses": poses,
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
            }

    def geometry(self):
        """Export compiled visual geometry; collision meshes stay in MuJoCo."""
        with self.lock:
            if self._geometry is None:
                self._geometry = self._export_geometry()
            return {**self._geometry, "state": self.snapshot()}

    def _export_geometry(self):
        model, meshes, geoms = self.model, {}, []
        for geom in self.visual_ids:
            mesh = int(model.geom_dataid[geom])
            kind = int(model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH) and str(mesh) not in meshes:
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                face, nface = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                meshes[str(mesh)] = {
                    "vertices": base64.b64encode(
                        np.asarray(model.mesh_vert[start : start + count], dtype="<f4").tobytes()
                    ).decode("ascii"),
                    "faces": base64.b64encode(
                        np.asarray(model.mesh_face[face : face + nface], dtype="<u4").tobytes()
                    ).decode("ascii"),
                }
            material = int(model.geom_matid[geom])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            geoms.append(
                {
                    "type": kind,
                    "mesh": mesh,
                    "size": model.geom_size[geom].tolist(),
                    "rgba": rgba.tolist(),
                    "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                }
            )
        return {"meshes": meshes, "geoms": geoms, "source": SOURCE}

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
