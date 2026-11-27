"""Local, non-pickle episode archives with aligned observations and actions."""

import io
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
from PIL import Image

from .assets import REVISION
from .environment import CAMERAS, HEIGHT, WIDTH


class Recording:
    def __init__(self, directory, env, *, seed, images):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".recording-", dir=self.directory)
        self.root = Path(self.temporary.name)
        self.id, self.images = str(uuid4()), images
        self.actions, self.rewards, self.terminated, self.truncated, self.sources = (
            [],
            [],
            [],
            [],
            [],
        )
        self.frames = []
        self.manifest = {
            "schema": "panda-episode-v1",
            "id": self.id,
            "created": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "environment": "PandaGraspPlace-v1",
            "origin": "simulation",
            "instruction": f"Grasp the {env.object['name']}, place it on the marked target, and withdraw.",
            "model_sha256": env.model_hash,
            "panda_revision": REVISION,
            "object": env.object,
            "mujoco": mujoco.__version__,
            "control_dt": env.dt,
            "physics_dt": float(env.model.opt.timestep),
            "max_steps": env.max_steps,
            "rgb": images,
            "action_spec": {
                "names": [f"joint{i}_target" for i in range(1, 8)] + ["finger_opening"],
                "units": ["rad"] * 7 + ["m"],
                "representation": "absolute position targets",
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
            },
            "proprio_spec": [
                "qpos[0:9]",
                "qvel[0:9]",
                "grasp_center_xyz_world",
                "grasp_center_quat_xyzw_world",
            ],
            "cameras": {},
            "pick": env.pick.tolist(),
            "orientation": env.orientation.tolist(),
        }
        for name in CAMERAS:
            fovy = float(env.model.cam_fovy[env.model.camera(name).id])
            focal = HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))
            self.manifest["cameras"][name] = {
                "width": WIDTH,
                "height": HEIGHT,
                "fovy": fovy,
                "K": [[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]],
                "convention": "camera-to-world; OpenGL -Z forward, +Y up",
                "renderer": "MuJoCo RGB",
            }
        try:
            self.capture(env)
        except Exception:
            self.temporary.cleanup()
            raise

    def capture(self, env):
        state, observation = env.capture_state(), env.observe()
        camera_poses = []
        for name in CAMERAS:
            camera = env.model.camera(name).id
            transform = np.eye(4)
            transform[:3, :3] = env.data.cam_xmat[camera].reshape(3, 3)
            transform[:3, 3] = env.data.cam_xpos[camera]
            camera_poses.append(transform)
        state.update(
            proprio=observation["proprio"],
            camera_poses=np.stack(camera_poses),
            qpos=env.data.qpos.copy(),
            qvel=env.data.qvel.copy(),
            time=np.array(env.steps * env.dt),
        )
        if self.images:
            for name, rgb in env.render_cameras().items():
                folder = self.root / "rgb" / name
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(folder / f"{len(self.frames):05d}.png", compress_level=1)
        self.frames.append(state)

    def append(self, env, action, result, source):
        _, reward, terminated, truncated, _ = result
        self.capture(env)
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.sources.append(source)

    def save(self, env, reason):
        if not self.actions:
            raise ValueError("No transitions recorded yet; take a step before saving")
        arrays = {key: np.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}
        arrays.update(
            actions=np.stack(self.actions),
            rewards=np.asarray(self.rewards),
            terminated=np.asarray(self.terminated),
            truncated=np.asarray(self.truncated),
            sources=np.asarray(self.sources),
        )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        self.manifest.update(
            transitions=len(self.actions),
            observations=len(self.frames),
            duration=(len(self.actions) * env.dt),
            end_reason=reason,
            outcome="success"
            if self.frames[-1]["aux"][2] >= 6
            else "failed"
            if self.frames[-1]["ended"][0]
            else "incomplete",
        )
        target = self.directory / f"{self.id}.zip"
        pending = target.with_suffix(".pending")
        with zipfile.ZipFile(
            pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            archive.writestr("episode.json", json.dumps(self.manifest, indent=2))
            archive.writestr("trajectory.npz", buffer.getvalue())
            for image in sorted(self.root.glob("rgb/*/*.png")):
                archive.write(image, image.relative_to(self.root).as_posix())
        pending.replace(target)
        self.temporary.cleanup()
        return dict(self.manifest)

    def close(self):
        self.temporary.cleanup()


class Episode:
    def __init__(self, path, env):
        with zipfile.ZipFile(path) as archive:
            if (
                archive.getinfo("episode.json").file_size > 100_000
                or archive.getinfo("trajectory.npz").file_size > 20_000_000
            ):
                raise ValueError("Episode exceeds supported size")
            self.manifest = json.loads(archive.read("episode.json"))
            if (
                self.manifest.get("schema") != "panda-episode-v1"
                or self.manifest.get("model_sha256") != env.model_hash
            ):
                raise ValueError("Episode does not match this Panda environment")
            if self.manifest.get("mujoco") != mujoco.__version__:
                raise ValueError("Episode requires the recorded MuJoCo version")
            count = self.manifest.get("transitions")
            if type(count) is not int or not 1 <= count <= env.max_steps:
                raise ValueError("Invalid episode length")
            with np.load(io.BytesIO(archive.read("trajectory.npz")), allow_pickle=False) as data:
                self.data = {key: data[key] for key in data.files}
        required = {
            "integration",
            "aux",
            "pick",
            "orientation",
            "previous_action",
            "ended",
            "qpos",
            "qvel",
            "proprio",
            "camera_poses",
            "time",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "sources",
        }
        if set(self.data) != required:
            raise ValueError("Episode fields do not match the recording schema")
        for key, values in self.data.items():
            expected = (
                count
                if key in {"actions", "rewards", "terminated", "truncated", "sources"}
                else count + 1
            )
            if len(values) != expected or (key != "sources" and not np.all(np.isfinite(values))):
                raise ValueError(f"Unaligned or nonfinite episode field: {key}")
        dimensions = {
            **{key: value.shape for key, value in env.capture_state().items()},
            "qpos": (env.model.nq,),
            "qvel": (env.model.nv,),
            "proprio": (25,),
            "camera_poses": (len(CAMERAS), 4, 4),
            "time": (),
            "actions": (8,),
            "rewards": (),
            "terminated": (),
            "truncated": (),
            "sources": (),
        }
        for key, shape in dimensions.items():
            if self.data[key].shape[1:] != shape:
                raise ValueError(f"Invalid episode dimensions: {key}")
        if not np.allclose(np.diff(self.data["time"]), env.dt, atol=1e-10, rtol=0):
            raise ValueError("Episode timestamps are not aligned with control steps")
        self.count = count

    def state(self, index):
        if type(index) is not int or not 0 <= index <= self.count:
            raise ValueError("Replay frame is out of range")
        return {
            key: self.data[key][index]
            for key in ("integration", "aux", "pick", "orientation", "previous_action", "ended")
        }
