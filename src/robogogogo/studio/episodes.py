"""robot episode lifecycle, aligned actions, and dataset persistence."""

from dataclasses import fields
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from robogogogo.data.store import EpisodeStore
from robogogogo.robot.cameras import mounted_rig

from .. import spatial as sp
from .schemas import CameraObservation, CameraSpec, NewEpisode, PolicyAction

if TYPE_CHECKING:
    from .workspace import Studio


class EpisodeService:
    """Serialize language-conditioned episode operations within one workspace."""

    def __init__(self, workspace: "Studio"):
        self.workspace = workspace
        self.store = EpisodeStore(workspace.directory / "episodes")

    def start_episode(self, request: NewEpisode):
        workspace = self.workspace
        with workspace.lock:
            if workspace.scene == "drop":
                raise ValueError("The episode workflow requires an articulated arm")
            instruction = request.instruction.strip()
            if not instruction:
                raise ValueError("A language instruction is required")
            cameras = None
            if request.camera_preset and request.cameras is not None:
                raise ValueError("Choose cameras or camera_preset, not both")
            specifications = request.cameras
            if request.camera_preset:
                specifications = [
                    CameraSpec(**config)
                    for config in mounted_rig(
                        workspace.session.model,
                        workspace.session.arms,
                        wrists=request.camera_preset == "head_wrists",
                    )
                ]
            if specifications is not None:
                cameras = [camera.model_dump(mode="json") for camera in specifications]
                if len({camera["name"] for camera in cameras}) != len(cameras):
                    raise ValueError("Camera names must be unique")
                for camera in cameras:
                    mount = camera["mount"]
                    if mount is not None and mount not in workspace.session.model.names:
                        raise ValueError(f"Unknown camera mount: {mount}")
            workspace.running = False
            workspace.task = None
            workspace.session.reset()
            delta = np.zeros_like(workspace.session.model.home)
            variants = {"a": [0.35, 0.15, -0.2], "b": [-0.35, -0.15, 0.25], "c": [0.15, -0.3, 0.35]}
            for arm in workspace.session.arms:
                indices = list(arm.joint_indices[:3])
                delta[indices] = variants[request.variant][: len(indices)]
            desired = jnp.clip(
                workspace.session.model.home + delta,
                workspace.session.model.lower,
                workspace.session.model.upper,
            )
            workspace.session.targets = {
                arm.name: np.asarray(sp.point_position(workspace.session.model, desired, arm.body))
                for arm in workspace.session.arms
            }
            workspace.session.restart_recording()
            workspace.episode = workspace._episode_for_state(instruction, cameras)
            workspace.epoch += 1
            workspace.error = None
            workspace.message = (
                "robot episode ready. Capture an observation, then execute an action."
            )
            workspace._refresh()
            return workspace.cached

    def capture_observation(self, request: CameraObservation):
        workspace = self.workspace
        with workspace.lock:
            if not workspace.episode:
                raise ValueError("Start a robot episode first")
            if workspace.running:
                raise ValueError("Pause live dynamics before capturing an aligned observation")
            if request.sequence != workspace.sequence:
                raise ValueError(
                    "The robot state changed during capture. Capture the current frame again."
                )
            index = workspace.episode.observe(
                request.images if request.images is not None else request.image,
                workspace.cached,
                ground_truth_depth=request.ground_truth_depth,
            )
            workspace.message = (
                f"Observation {index:04d} captured: RGB + instruction + proprioception"
            )
            workspace._refresh()
            return dict(observation=index, snapshot=workspace.cached)

    def policy_action(self, request: PolicyAction, *, collision_world=None, task_step=False):
        workspace = self.workspace
        with workspace.lock:
            episode = workspace.episode
            if getattr(workspace, "task", None) is not None and not task_step:
                raise ValueError("Use the coordinated task executor or start a new episode")
            if not episode or episode.done:
                raise ValueError("Start a new episode before executing actions")
            if workspace.running:
                raise ValueError("Pause live dynamics before executing policy actions")
            if not episode.observations or request.observation != len(episode.observations) - 1:
                raise ValueError("Action requires the latest camera observation")
            observation = episode.observations[-1]
            s = workspace.session
            if observation["time"] != s.elapsed or not np.array_equal(
                observation["joint_position"], np.asarray(s.state.q)
            ):
                raise ValueError(
                    "Observation is stale. Capture the current robot state before acting."
                )
            delta = np.asarray(request.delta, dtype=float)
            if delta.shape != s.model.home.shape or not np.isfinite(delta).all():
                raise ValueError("Action delta must contain one finite value per joint")
            limit = np.asarray(episode.action_spec["max_delta"])
            desired = jnp.clip(
                s.state.q + jnp.asarray(np.clip(delta, -limit, limit)), s.model.lower, s.model.upper
            )
            if s.mobile is not None:
                s.mobile.validate_goal(np.asarray(s.state.q), np.asarray(desired))
            applied = np.asarray(desired - s.state.q)
            previous = workspace.cached
            s.target_q = desired  # Preserve the task's target separately from the controller goal.
            executed, failed = 0, False
            try:
                for _ in range(request.control_ticks):
                    if len(s.history) >= workspace.max_samples:
                        raise ValueError(
                            "Physics recording limit reached. Export and start a new episode."
                        )
                    s.advance(collision_world=collision_world)
                    executed += 1
                    if task_step and workspace.task_service.after_tick():
                        break
            except FloatingPointError as error:
                workspace.error, failed = str(error), True
            if failed and executed == 0:
                episode.invalidate(workspace.error or "Action rejected before state advance")
                if episode.actions:
                    episode.actions[-1]["truncated"] = True
                workspace.message = "Action rejected before state advance"
                workspace._refresh()
                return workspace.cached
            workspace._refresh()
            episode.append_action(
                requested=delta,
                applied=applied,
                ticks=executed,
                source=request.source,
                previous=previous,
                following=workspace.cached,
                failed=failed,
            )
            workspace.message = (
                f"Action {len(episode.actions):04d} executed · capture the next RGB observation"
            )
            if episode.done:
                workspace.message = (
                    f"Episode complete: {episode.reason}. Capture the final observation and export."
                )
            workspace._refresh()
            return workspace.cached

    def rule_action(self):
        from robogogogo.runtime.policies import ReachPolicy
        from robogogogo.runtime.supervisor import ControlSystem

        from ..contracts import Observation

        workspace = self.workspace
        with workspace.lock:
            episode = workspace.episode
            if getattr(workspace, "task", None) is not None:
                raise ValueError("Use the coordinated task executor or start a new episode")
            if not episode or episode.done:
                raise ValueError("Start a new episode first")
            session = workspace.session
            if getattr(self, "_control_model", None) is not session.model:
                self._control_model = session.model
                self._control_system = ControlSystem(
                    {"rule_based_reach": ReachPolicy(session.model, session.arms)},
                    lower=session.model.lower,
                    upper=session.model.upper,
                    max_delta=episode.action_spec["max_delta"],
                )
            observation = Observation(
                episode.instruction,
                b"",
                tuple(np.asarray(session.state.q)),
                tuple(np.asarray(session.state.qd)),
                session.elapsed,
                targets=episode.targets,
            )
            decision = self._control_system.step(observation, controller="rule_based_reach")
            return dict(
                delta=list(decision.action.actions[0]),
                source="rule_based",
                controller=decision.controller,
                limited=decision.limited,
                note="Position IK using privileged simulator targets; no collision avoidance",
            )

    def reference_action(self):
        workspace = self.workspace
        with workspace.lock:
            if not workspace.episode or workspace.episode.done:
                raise ValueError("Start a new episode first")
            q, _ = workspace.session.solve_targets(workspace.episode.targets)
            limit = np.asarray(workspace.episode.action_spec["max_delta"])
            return dict(
                delta=np.clip(np.asarray(q - workspace.session.state.q), -limit, limit).tolist(),
                source="scripted_reference",
                note="Damped-least-squares IK baseline; not a learned VLA policy",
            )

    def save_episode(self):
        workspace = self.workspace
        with workspace.lock:
            if not workspace.episode:
                raise ValueError("No episode to export")
            arrays = {
                f.name: np.asarray(getattr(workspace.session.model, f.name))
                for f in fields(workspace.session.model)
                if f.name not in ("parents", "kinds", "indices", "names")
            }
            return self.store.save(
                workspace.episode, model=workspace.model_info, model_arrays=arrays
            )

    def episodes(self):
        with self.workspace.lock:
            return self.store.list()
