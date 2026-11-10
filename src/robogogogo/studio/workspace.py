"""Simulation workspace lifecycle, state publication, and checkpoints."""

import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from uuid import UUID, uuid4

import jax.numpy as jnp
import numpy as np

from robogogogo.data.recording import Episode
from robogogogo.robot.cameras import mounted_rig

from .. import models, urdf
from .. import spatial as sp
from ..session import Session
from .episodes import EpisodeService
from .schemas import CameraSpec, Command, NewEpisode, Project


class Studio:
    """Thread-safe local workspace, independent of HTTP transport."""

    max_samples = 18001

    def __init__(self, directory: Path, *, scene="arm"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.running = False
        self.speed = 1.0
        self.sequence = 0
        self.epoch = 0
        self.project_id = None
        self.project_name = "Untitled experiment"
        self.message = "Ready. Set a joint goal or run the simulation."
        self.error = None
        self.episode = None
        self.task = None
        self.episode_service = EpisodeService(self)
        from .tasks import TaskService

        self.task_service = TaskService(self)
        self._run_wall = time.perf_counter()
        self._run_sim = 0.0
        self._replace_session(scene)
        if scene != "drop":
            self.episode_service.start_episode(NewEpisode(camera_preset="head_wrists"))

    def _replace_session(self, scene, xml=None, dt=0.002, substeps=5, arms=None):
        from ..robot.mobile_design import (
            DESIGN,
            mobile_arm_definitions,
            mobile_dual_arm_urdf,
            mobile_home,
        )

        if scene == "mobile_dual_arm":
            xml, arms = mobile_dual_arm_urdf(), mobile_arm_definitions()
        if scene in ("dual_arm", "dual_arm_head"):
            xml = models.multi_arm_urdf(2)
            arms = [
                {"name": "left_arm", "tool": "left_tool"},
                {"name": "right_arm", "tool": "right_tool"},
            ]
        if scene in ("urdf", "dual_arm", "dual_arm_head", "mobile_dual_arm"):
            if not xml:
                raise ValueError("URDF content is required")
            try:
                robot = urdf.loads(xml, dt=dt)
            except (ET.ParseError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid URDF: {error}") from error
            model, label = robot.model, robot.name
            visuals = list(robot.visuals)
            if scene == "mobile_dual_arm":
                model = replace(model, home=jnp.asarray(mobile_home(model)))
            if scene in ("dual_arm", "dual_arm_head"):
                model = replace(model, home=jnp.tile(models.arm6(dt=dt).home, 2))
        elif scene == "arm":
            arms = arms or [{"name": "arm", "tool": "tool"}]
            model, label = models.arm6(dt=dt), "RG-6 Research Arm"
            visuals = list(
                urdf.loads(files("robogogogo").joinpath("assets/arm6.urdf").read_text()).visuals
            )
        elif scene == "drop":
            model, label, visuals = models.slider(dt=dt), "Guided sphere", []
        else:
            raise ValueError("Unknown scene")
        if scene == "dual_arm_head":
            model, visuals = models.with_pan_tilt_head(model, visuals)
            label += " + Pan/Tilt Head"
        elif scene in ("arm", "dual_arm"):
            model, visuals = models.with_camera_mast(model, visuals)
        if model.home.size > 48 or len(model.parents) > 128:
            raise ValueError("Studio currently supports up to 48 moving joints and 128 links")
        candidate = Session(
            model,
            scene="drop" if scene == "drop" else "arm",
            substeps=substeps,
            arms=arms,
            mobile=scene == "mobile_dual_arm",
        )
        self.session, self.scene, self.xml = candidate, scene, xml
        self.episode = None
        self.task = None
        self.running = False
        self.epoch += 1
        self.error = None
        moving = [(i, index) for i, index in enumerate(model.indices) if index >= 0]

        def finite(value):
            return float(value) if np.isfinite(float(value)) else None

        self.model_info = dict(
            name=label,
            mobile_design=DESIGN if scene == "mobile_dual_arm" else None,
            scene=scene,
            dof=int(model.home.size),
            arms=[arm.to_dict() for arm in candidate.arms],
            camera_presets={
                key: [
                    CameraSpec(**config).model_dump(mode="json")
                    for config in mounted_rig(model, candidate.arms, wrists=wrists)
                ]
                for key, wrists in [("head_wrists", True), ("head", False)]
            }
            if "head" in model.names and len(candidate.arms) <= 6
            else {},
            names=list(model.names),
            parents=list(model.parents),
            kinds=list(model.kinds),
            indices=list(model.indices),
            masses=np.asarray(model.mass).tolist(),
            axes=np.asarray(model.axis).tolist(),
            joints=[
                dict(
                    name=model.names[i],
                    body=i,
                    kind=model.kinds[i],
                    lower=finite(model.lower[j]),
                    upper=finite(model.upper[j]),
                    effort=float(model.effort[j]),
                    home=float(model.home[j]),
                )
                for i, j in moving
            ],
            dt=float(model.dt),
            substeps=substeps,
            visuals=visuals,
            spheres=[
                dict(body=int(b), center=np.asarray(c).tolist(), radius=float(r))
                for b, c, r in zip(model.sphere_body, model.sphere_center, model.sphere_radius)
            ],
            backend="JAX / CPU"
            if candidate.state.q.device.platform == "cpu"
            else f"JAX / {candidate.state.q.device.platform.upper()}",
            dtype=str(candidate.state.q.dtype),
            max_samples=self.max_samples,
        )
        self._run_wall, self._run_sim = time.perf_counter(), 0.0
        self._refresh()

    def _refresh(self):
        s, m = self.session, self.session.model
        pose = s.poses()
        self.sequence += 1
        contact_count, normal_force, penetration = 0, 0.0, 0.0
        if m.sphere_radius.size:
            contacts, _ = sp.plane_contacts(m, s.state)
            contact_count = int(jnp.sum(contacts.force[:, 2] > 0))
            normal_force = float(jnp.sum(contacts.force[:, 2]))
            penetration = float(jnp.max(contacts.penetration))
        from ..simulation.materials import material_catalog

        self.cached = dict(
            object_materials=material_catalog(),
            sequence=self.sequence,
            epoch=self.epoch,
            model=self.model_info,
            episode=self.episode.summary() if self.episode else None,
            coordination=self.task_service.summary() if self.task else None,
            manipulation=self.task_service.manipulation.summary()
            if self.task and self.task_service.manipulation
            else None,
            project=dict(id=self.project_id, name=self.project_name),
            state=dict(
                mobile=self.session.mobile_status,
                q=np.asarray(s.state.q).tolist(),
                qd=np.asarray(s.state.qd).tolist(),
                goal=np.asarray(s.target_q).tolist(),
                target=np.asarray(s.target).tolist(),
                targets={name: value.tolist() for name, value in s.targets.items()},
                tool_positions={
                    arm.name: np.asarray(pose.position[arm.body]).tolist() for arm in s.arms
                },
                target_errors={
                    arm.name: float(
                        np.linalg.norm(np.asarray(pose.position[arm.body]) - s.targets[arm.name])
                    )
                    for arm in s.arms
                },
                torque=s.last_torque.tolist(),
                positions=np.asarray(pose.position).tolist(),
                rotations=np.asarray(pose.rotation).tolist(),
                time=s.elapsed,
                energy=s.mechanical_energy(),
                distance=max(
                    float(np.linalg.norm(np.asarray(pose.position[arm.body]) - s.targets[arm.name]))
                    for arm in s.arms
                ),
                running=self.running,
                speed=self.speed,
                samples=len(s.history),
                rtf=(s.elapsed - self._run_sim) / max(time.perf_counter() - self._run_wall, 1e-6)
                if self.running
                else 0.0,
                contacts=contact_count,
                normal_force=normal_force,
                penetration=penetration,
                message=self.message,
                error=self.error,
            ),
        )

    def snapshot(self):
        with self.lock:
            return self.cached

    def _advance(self, count):
        for _ in range(count):
            if len(self.session.history) >= self.max_samples:
                self.running = False
                self.message = "Recording limit reached. Export your trajectory, then reset to start a new run."
                break
            self.session.advance()

    def tick(self):
        with self.lock:
            if not self.running:
                return
            try:
                self._advance(2)
            except FloatingPointError as error:
                self.running = False
                self.error = str(error)
            self._refresh()

    def command(self, command: Command):
        with self.lock:
            op = command.op
            if self.task is not None and op not in ("pause", "speed", "scene", "reset"):
                raise ValueError(
                    "Coordinated task owns the robot. Start a new episode to use debug controls."
                )
            if self.task is not None and op == "pause":
                self.task.pause()
            if op == "reset":
                self.task = None
            if op == "run":
                if self.error:
                    raise ValueError("Reset the failed simulation before running")
                if len(self.session.history) >= self.max_samples:
                    raise ValueError("Export the recording and reset before running again")
                self.running = True
                self._run_wall, self._run_sim = time.perf_counter(), self.session.elapsed
                self.message = "Simulation running"
            elif op == "pause":
                self.session.stop_drive()
                self.running = False
                self.message = "Simulation paused"
            elif op == "step":
                self.running = False
                try:
                    self._advance(1)
                except FloatingPointError as error:
                    self.error = str(error)
                    self._refresh()
                    return self.cached
                if len(self.session.history) < self.max_samples:
                    self.message = "Advanced one control tick"
            elif op == "reset":
                self.running = False
                self.session.reset()
                self.epoch += 1
                self.error = None
                self.message = "Reset to home. A new recording has started."
            elif op == "drive":
                if command.values is None:
                    raise ValueError("Drive velocities are required")
                self.session.set_drive(
                    command.values, duration=command.duration, friction=command.friction
                )
                self.message = "Wheel drive goal set; Run to move. Lease uses simulation time."
            elif op == "brake":
                if self.session.mobile is None:
                    raise ValueError("Wheel braking requires the mobile scene")
                self.session.stop_drive()
                self.message = "Wheel speed target zero; friction-limited braking while running"
            elif op == "home":
                q = np.asarray(self.session.model.home).copy()
                if self.session.mobile is not None:
                    q[self.session.mobile.driven] = np.asarray(self.session.state.q)[
                        self.session.mobile.driven
                    ]
                self.session.set_joints(q)
                self.message = "Home goal applied. Run to move the robot."
            elif op == "joints":
                if command.values is None:
                    raise ValueError("Joint goals are required")
                self.session.set_joints(command.values, command.arm)
                self.message = "Joint goals applied"
            elif op == "posture":
                if self.scene != "mobile_dual_arm" or command.posture is None:
                    raise ValueError("A mobile concept scene and named posture are required")
                from ..robot.mobile_design import mobile_home

                q = np.asarray(self.session.target_q).copy()
                preset = mobile_home(self.session.model, command.posture)
                for name in ("torso_lower", "torso_upper", "torso"):
                    index = self.session.model.indices[self.session.model.names.index(name)]
                    q[index] = preset[index]
                self.session.set_joints(q)
                self.message = "Torso goal passed static support guard. Run to move with dynamic margin monitoring; collision avoidance is not implemented."
            elif op == "target":
                if self.scene == "drop":
                    raise ValueError("Cartesian control is unavailable in the contact experiment")
                if command.values is None:
                    raise ValueError("Target coordinates are required")
                residual = self.session.set_target(command.values, command.arm)
                self.message = f"Position IK residual: {residual * 1000:.2f} mm" + (
                    " — target may be unreachable" if residual > 0.02 else ""
                )
            elif op == "scene":
                if command.scene is None:
                    raise ValueError("Scene is required")
                self._replace_session(command.scene)
                self.project_id, self.project_name = None, "Untitled experiment"
                if command.scene != "drop":
                    self.episode_service.start_episode(NewEpisode(camera_preset="head_wrists"))
                self.message = "Embodiment loaded. New episode ready."
            elif op == "speed":
                if command.speed is None:
                    raise ValueError("Speed is required")
                self.speed = command.speed
                self._run_wall, self._run_sim = time.perf_counter(), self.session.elapsed
            if self.episode and op not in ("pause", "speed", "scene"):
                self.episode.invalidate("State changed outside the recorded VLA action loop")
            self._refresh()
            return self.cached

    def _episode_for_state(self, instruction, cameras=None):
        m = self.session.model
        moving = [(name, kind) for name, kind, i in zip(m.names, m.kinds, m.indices) if i >= 0]
        if cameras is not None:
            cameras = [dict(camera) for camera in cameras]
            for camera in cameras:
                mount = camera.get("mount")
                if mount is not None and mount not in m.names:
                    raise ValueError(f"Unknown camera mount: {mount}")
                camera["body"] = m.names.index(mount) if mount is not None else None
                focal = camera["height"] / (
                    2 * np.tan(np.deg2rad(camera["vertical_fov_degrees"]) / 2)
                )
                camera["intrinsics"] = [
                    [focal, 0, camera["width"] / 2],
                    [0, focal, camera["height"] / 2],
                    [0, 0, 1],
                ]
                camera.update(encoding="png", channels=3, color_space="sRGB")
        return Episode(
            instruction,
            self.session.target,
            cameras=cameras,
            arms=[arm.to_dict() for arm in self.session.arms],
            targets=self.session.targets,
            joint_names=[p[0] for p in moving],
            joint_kinds=[p[1] for p in moving],
            dt=float(m.dt),
            substeps=self.session.substeps,
        )

    def import_robot(self, xml, arms=None):
        with self.lock:
            self._replace_session("urdf", xml, arms=arms)
            self.project_id, self.project_name = None, "Imported robot"
            self.episode = self._episode_for_state("Move the robot tool to the highlighted target.")
            self.message = "URDF loaded. Fixed-base dynamics; supported primitive visuals."
            self._refresh()
            return self.cached

    def _project_path(self, identifier):
        try:
            identifier = str(UUID(identifier))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Invalid project ID") from error
        return self.directory / f"{identifier}.json"

    def projects(self):
        with self.lock:
            result = []
            for path in self.directory.glob("*.json"):
                try:
                    p = Project.model_validate_json(path.read_text())
                    result.append(
                        dict(id=p.id, name=p.name, updated=p.updated, scene=p.scene, time=p.elapsed)
                    )
                except (ValueError, OSError):
                    continue
            return sorted(result, key=lambda p: p["updated"], reverse=True)

    def save(self, request):
        with self.lock:
            name = request.name.strip()
            if not name:
                raise ValueError("Project name cannot be blank")
            identifier = request.id or str(uuid4())
            path = self._project_path(identifier)
            if request.id and not path.exists():
                raise FileNotFoundError("Project not found")
            s = self.session
            project = Project(
                id=identifier,
                name=name,
                updated=datetime.now(timezone.utc).isoformat(),
                scene=self.scene,
                xml=self.xml,
                dt=float(s.model.dt),
                substeps=s.substeps,
                q=np.asarray(s.state.q).tolist(),
                qd=np.asarray(s.state.qd).tolist(),
                goal=np.asarray(s.target_q).tolist(),
                target=np.asarray(s.target).tolist(),
                elapsed=s.elapsed,
                tire_friction=s.tire_mu,
                mobile_com_acceleration=np.asarray(s.com_acceleration).tolist(),
                arms=[{"name": arm.name, "tool": arm.tool, "root": arm.root} for arm in s.arms],
                targets={name: value.tolist() for name, value in s.targets.items()},
                cameras=[
                    {key: value for key, value in c.items() if key in CameraSpec.model_fields}
                    for c in self.episode.cameras
                ]
                if self.episode and self.episode.multiview
                else None,
                instruction=self.episode.instruction
                if self.episode
                else "Move the robot tool to the highlighted target.",
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(project.model_dump_json(indent=2))
            temporary.replace(path)
            self.project_id, self.project_name = identifier, name
            self.message = "Experiment checkpoint saved locally"
            self._refresh()
            return dict(project=project.model_dump(), snapshot=self.cached)

    def load(self, identifier):
        with self.lock:
            project = Project.model_validate_json(self._project_path(identifier).read_text())
            # Validate vectors before replacing the active workspace.
            expected = len(project.q)
            if not expected or len(project.qd) != expected or len(project.goal) != expected:
                raise ValueError("Invalid project state dimensions")
            if any(abs(x) > 1e6 for x in project.q + project.qd + project.goal + project.target):
                raise ValueError("Project state is outside supported numeric bounds")
            old = dict(self.__dict__)
            try:
                self._replace_session(
                    project.scene,
                    project.xml,
                    project.dt,
                    project.substeps,
                    [arm.model_dump() for arm in project.arms] if project.arms else None,
                )
                if expected != self.session.model.home.size:
                    raise ValueError("Project dimensions do not match the robot")
                s = self.session
                s.state = sp.State(jnp.asarray(project.q), jnp.asarray(project.qd))
                s.target_q, s.target = jnp.asarray(project.goal), np.asarray(project.target)
                if project.targets is not None:
                    if set(project.targets) != set(s.targets):
                        raise ValueError("Checkpoint target names do not match arm groups")
                    s.targets = {name: np.asarray(value) for name, value in project.targets.items()}
                s.elapsed = project.elapsed
                s.tire_mu = project.tire_friction
                s.com_acceleration = jnp.asarray(project.mobile_com_acceleration)
                if s.mobile is not None:
                    s.mobile.validate_goal(np.asarray(s.state.q), np.asarray(s.target_q))
                s.restart_recording()
                if self.scene != "drop":
                    self.episode = self._episode_for_state(
                        project.instruction,
                        [c.model_dump(mode="json") for c in project.cameras]
                        if project.cameras
                        else None,
                    )
                self.project_id, self.project_name = project.id, project.name
                self.message = (
                    "Checkpoint restored. Paused; a new trajectory recording has started."
                )
                self._refresh()
                return self.cached
            except Exception:
                self.__dict__.update(old)
                raise

    def export(self):
        with self.lock:
            handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
            path = Path(handle.name)
            handle.close()
            try:
                self.session.export(path)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            return path
