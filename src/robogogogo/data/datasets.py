"""Read aligned robogogogo episodes for training, evaluation, or offline RL.

Images are returned as PNG bytes so downstream adapters choose their own image
libraries/preprocessing. This reader never unpickles or extracts archive paths.
"""

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from robogogogo.vla import Observation


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: tuple[float, ...]
    next_observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    source: str


def iter_transitions(path: str | Path, *, include_ground_truth=False) -> Iterator[Transition]:
    """Yield only fully aligned transitions; reject incomplete or malformed data.

    Training callers can deliberately exclude partial episodes before calling.
    No implicit resizing, normalization, action conversion or source relabeling.
    """
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo("episode.json").file_size > 20_000_000:
            raise ValueError("Episode manifest is oversized")
        manifest = json.loads(archive.read("episode.json"))
        if manifest.get("format") not in ("robogogogo-episode-v1", "robogogogo-episode-v2"):
            raise ValueError("Unsupported episode format")
        if manifest.get("action_spec", {}).get("representation") != "joint_position_delta":
            raise ValueError("Unsupported action representation")
        observations = manifest["observations"]

        def observation(index):
            if not isinstance(index, int) or index < 0 or index >= len(observations):
                raise ValueError("Transition has no aligned camera observation")
            data = observations[index]
            cameras = manifest.get("cameras", [manifest.get("camera", {})])
            images = {}
            for camera in cameras:
                camera_name = camera["name"]
                name = (
                    f"frames/{camera_name}/{index:06d}.png"
                    if manifest["format"].endswith("v2")
                    else f"frames/{index:06d}.png"
                )
                reference = data.get("images", {camera_name: data.get("image")}).get(camera_name)
                if reference != name or archive.getinfo(name).file_size > 512_000:
                    raise ValueError("Invalid camera frame reference")
                images[camera_name] = archive.read(name)
            depths = {}
            if include_ground_truth:
                for camera in cameras:
                    name = f"depth/{camera['name']}/{index:06d}.npy"
                    if name in archive.namelist():
                        if archive.getinfo(name).file_size > 1_300_000:
                            raise ValueError("Depth frame is oversized")
                        depth = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                        if (
                            depth.shape != (camera["height"], camera["width"])
                            or depth.dtype != np.float32
                        ):
                            raise ValueError("Invalid depth frame")
                        depths[camera["name"]] = depth
            q, qd = tuple(data["joint_position"]), tuple(data["joint_velocity"])
            if len(q) != manifest["action_spec"]["dimensions"] or len(qd) != len(q):
                raise ValueError("Observation dimensions do not match action contract")
            if not np.isfinite(q + qd + (data["time"],)).all():
                raise ValueError("Nonfinite state in episode")
            return Observation(
                data["instruction"],
                images[cameras[0]["name"]],
                q,
                qd,
                data["time"],
                images,
                tuple(cameras),
                data.get("camera_poses", {}),
                tuple(manifest.get("arms", [])),
                data.get("tool_positions", {}),
                data.get("targets", {}),
                depths,
                data.get("manipulation"),
            )

        for transition in manifest["transitions"]:
            before = observation(transition["observation"])
            after = observation(transition["next_observation"])
            action = tuple(transition["applied_delta"])
            if (
                before.time != transition["time"]
                or after.time != transition["next_time"]
                or after.time <= before.time
            ):
                raise ValueError("Transition timestamps are not aligned")
            if (
                len(action) != len(before.joint_position)
                or not np.isfinite(action + (transition["reward"],)).all()
            ):
                raise ValueError("Invalid action or reward")
            yield Transition(
                before,
                action,
                after,
                float(transition["reward"]),
                bool(transition["terminated"]),
                bool(transition["truncated"]),
                transition["source"],
            )
