"""Aligned, source-labeled visual episode recording and export."""

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..perception.depth import decode_depth
from .cameras import world_camera_pose
from .contracts import CAMERA
from .images import decode_camera_png


class Episode:
    max_observations = 501

    def __init__(
        self,
        instruction,
        target,
        *,
        joint_names,
        joint_kinds,
        dt,
        substeps,
        horizon=250,
        cameras=None,
        arms=None,
        targets=None,
    ):
        self.cameras = [dict(camera) for camera in (cameras or [CAMERA])]
        self.arms = list(arms or [])
        self.targets = {
            name: np.asarray(value, dtype=float).tolist() for name, value in (targets or {}).items()
        }
        self.multiview = cameras is not None or len(self.arms) > 1
        self.id = str(uuid4())
        self.instruction = instruction
        self.target = np.asarray(target, dtype=float).tolist()
        self.created = datetime.now(timezone.utc).isoformat()
        self.joint_names = list(joint_names)
        self.joint_kinds = list(joint_kinds)
        self.dt, self.substeps, self.horizon = dt, substeps, horizon
        self.observations = []
        self.images = []
        self.depths = []
        self.actions = []
        self.done = False
        self.success = False
        self.reason = None
        self.invalidated = False
        self.default_control_ticks = 10
        self.task_plan = None
        self.task_progress = None

    @property
    def action_spec(self):
        return dict(
            representation="joint_position_delta",
            dimensions=len(self.joint_names),
            arms=self.arms,
            names=self.joint_names,
            units=["m" if k == "prismatic" else "rad" for k in self.joint_kinds],
            max_delta=[0.02 if k == "prismatic" else 0.12 for k in self.joint_kinds],
            default_control_ticks=self.default_control_ticks,
            physics_dt=self.dt,
            physics_substeps=self.substeps,
            gripper=bool(
                self.task_plan and self.task_plan.get("kind") in ("handover", "material_grasp")
            ),
            gripper_control="task_stage_events"
            if self.task_plan and self.task_plan.get("kind") in ("handover", "material_grasp")
            else None,
            frame="joint_coordinates",
            normalization="none",
        )

    def summary(self):
        return dict(
            id=self.id,
            instruction=self.instruction,
            task=self.task_plan["kind"]
            if self.task_plan
            else ("multi_arm_reach" if len(self.arms) > 1 else "reach_target"),
            targets=self.targets,
            arms=self.arms,
            target=self.target,
            observations=len(self.observations),
            actions=len(self.actions),
            horizon=self.horizon,
            done=self.done,
            success=self.success,
            reason=self.reason,
            camera=self.cameras[0],
            cameras=self.cameras,
            action_spec=self.action_spec,
            task_plan=self.task_plan,
            task_progress=self.task_progress,
        )

    def observe(self, data_url, snapshot, *, ground_truth_depth=None):
        if self.invalidated:
            raise ValueError(
                "Robot state changed outside the episode action loop. Export existing data or start a new episode."
            )
        if len(self.observations) >= self.max_observations:
            raise ValueError("Episode observation limit reached. Export and start a new episode.")
        state = snapshot["state"]
        if isinstance(data_url, str):
            if len(self.cameras) != 1:
                raise ValueError("Capture every configured camera in one observation")
            data_url = {self.cameras[0]["name"]: data_url}
        if set(data_url) != {camera["name"] for camera in self.cameras}:
            raise ValueError("Observation must contain exactly the configured camera names")
        image = {
            camera["name"]: decode_camera_png(data_url[camera["name"]], camera)
            for camera in self.cameras
        }
        depths = {}
        if ground_truth_depth is not None:
            if set(ground_truth_depth) != set(image):
                raise ValueError("Depth bundle must include every configured camera")
            depths = {
                c["name"]: decode_depth(ground_truth_depth[c["name"]], c) for c in self.cameras
            }
        # Repeated capture at the same state updates the RGB, not the transition index.
        if (
            self.observations
            and self.observations[-1]["time"] == state["time"]
            and self.observations[-1]["joint_position"] == state["q"]
        ):
            self.images[-1] = image
            self.depths[-1] = depths
            self.observations[-1]["sequence"] = snapshot["sequence"]
            return len(self.observations) - 1
        index = len(self.observations)
        paths = {
            camera["name"]: (
                f"frames/{camera['name']}/{index:06d}.png"
                if self.multiview
                else f"frames/{index:06d}.png"
            )
            for camera in self.cameras
        }
        observation = dict(
            index=index,
            sequence=snapshot["sequence"],
            time=state["time"],
            instruction=self.instruction,
            image=paths[self.cameras[0]["name"]],
            images=paths,
            camera_poses={
                camera["name"]: world_camera_pose(camera, state) for camera in self.cameras
            },
            manipulation=snapshot.get("manipulation"),
            joint_position=state["q"],
            joint_velocity=state["qd"],
            tool_position=state["positions"][self.arms[0]["body"]]
            if self.arms
            else state["positions"][-1],
            tool_positions={arm["name"]: state["positions"][arm["body"]] for arm in self.arms},
            targets=self.targets,
            target=list(self.target),
        )
        if self.actions and self.actions[-1]["next_observation"] is None:
            if abs(state["time"] - self.actions[-1]["next_time"]) < 1e-8:
                self.actions[-1]["next_observation"] = index
        self.observations.append(observation)
        self.images.append(image)
        self.depths.append(depths)
        return index

    def invalidate(self, reason):
        self.invalidated = True
        if not self.done:
            self.done, self.reason = True, reason

    def append_action(
        self, *, requested, applied, ticks, source, previous, following, failed=False
    ):
        errors = {
            arm["name"]: float(
                np.linalg.norm(
                    np.asarray(following["state"]["positions"][arm["body"]])
                    - self.targets[arm["name"]]
                )
            )
            for arm in self.arms
        }
        distance = (
            max(errors.values())
            if errors
            else float(
                np.linalg.norm(np.asarray(following["state"]["positions"][-1]) - self.target)
            )
        )
        self.success = bool(
            self.task_plan is None
            and not failed
            and distance < 0.02
            and np.linalg.norm(following["state"]["qd"]) < 0.1
        )
        self.done = failed or self.success or len(self.actions) + 1 >= self.horizon
        self.reason = (
            "numerical_failure"
            if failed
            else "success"
            if self.success
            else "horizon"
            if self.done
            else None
        )
        self.actions.append(
            dict(
                index=len(self.actions),
                observation=len(self.observations) - 1,
                next_observation=None,
                requested_delta=np.asarray(requested, dtype=float).tolist(),
                applied_delta=np.asarray(applied, dtype=float).tolist(),
                control_ticks=ticks,
                source=source,
                time=previous["state"]["time"],
                next_time=following["state"]["time"],
                next_joint_position=following["state"]["q"],
                next_joint_velocity=following["state"]["qd"],
                reward=-sum(value**2 for value in errors.values()) if errors else -(distance**2),
                target_errors=errors,
                success=self.success,
                terminated=failed or self.success,
                truncated=self.done and not (failed or self.success),
            )
        )

    def export(self, destination, *, model, model_arrays):
        if not self.observations:
            raise ValueError("Capture an observation before exporting an episode")
        manifest = dict(
            format="robogogogo-episode-v2" if self.multiview else "robogogogo-episode-v1",
            camera_convention="position/look_at/up in mount coordinates; null mount is world; optical +X right, +Y down, +Z forward",
            id=self.id,
            created=self.created,
            instruction=self.instruction,
            task=self.task_plan["kind"]
            if self.task_plan
            else ("multi_arm_reach" if len(self.arms) > 1 else "reach_target"),
            targets=self.targets,
            arms=self.arms,
            target=self.target,
            camera=self.cameras[0],
            cameras=self.cameras,
            action_spec=self.action_spec,
            task_plan=self.task_plan,
            task_progress=self.task_progress,
            model=model,
            observations=self.observations,
            transitions=self.actions,
            success=self.success,
            finished=self.done,
            end_reason=self.reason,
            depth_contract=dict(
                encoding="npy-float32",
                units="m",
                quantity="optical_z",
                invalid="NaN",
                source="renderer_ground_truth",
                privileged=True,
                paths="depth/{camera}/{observation:06d}.npy",
            ),
            rendering="Browser camera required; no headless RGB renderer in this version",
            sources=sorted({a["source"] for a in self.actions}),
            missing_next_observations=sum(a["next_observation"] is None for a in self.actions),
        )
        import io

        physics = io.BytesIO()
        np.savez_compressed(physics, **model_arrays)
        path = Path(destination)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("episode.json", json.dumps(manifest, indent=2, allow_nan=False))
            archive.writestr("physics.npz", physics.getvalue())
            for i, bundle in enumerate(self.depths):
                for name, depth in bundle.items():
                    buffer = io.BytesIO()
                    np.save(buffer, depth, allow_pickle=False)
                    archive.writestr(f"depth/{name}/{i:06d}.npy", buffer.getvalue())
            for i, image in enumerate(self.images):
                for name, data in image.items():
                    archive.writestr(self.observations[i]["images"][name], data)
        return path
