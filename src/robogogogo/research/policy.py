"""Small visual nearest-neighbor imitation baseline and external trainer contract.

This is not a VLA model. External trainers receive episode paths, output_dir and
seed and return a policy with reset() and act(observation, instruction) methods.
Only rgb, proprio and previous_action are exposed to a learned policy.
"""

from pathlib import Path
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: dict, instruction: str) -> np.ndarray: ...


def features(rgb, proprio, previous_action):
    """8x8 RGB block means plus proprioception and the last requested action."""
    rgb = np.asarray(rgb)
    height, width = rgb.shape[-3:-1]
    blocks = rgb.reshape(*rgb.shape[:-3], 8, height // 8, 8, width // 8, 3)
    image = blocks.mean(axis=(-4, -2)).reshape(*rgb.shape[:-3], 192) / 255.0
    return np.concatenate((image, proprio, previous_action), axis=-1).astype(np.float32)


class NearestPolicy:
    def __init__(self, x, y, mean, scale):
        self.x, self.y, self.mean, self.scale = x, y, mean, scale

    def reset(self):
        pass

    def act(self, observation, instruction):
        query = features(**observation)
        query = (query - self.mean) / self.scale
        # Small chunks avoid a temporary proportional to the entire dataset.
        best_distance, best_index = float("inf"), 0
        for start in range(0, len(self.x), 2048):
            delta = self.x[start : start + 2048] - query
            distances = np.einsum("ij,ij->i", delta, delta)
            index = int(np.argmin(distances))
            if distances[index] < best_distance:
                best_distance, best_index = float(distances[index]), start + index
        return self.y[best_index].copy()


def train_nearest(episodes, *, output_dir, seed):
    """Fit on every transition, including failed demonstrations; never cherry-pick."""
    x, y = [], []
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            x.append(features(data["rgb"][:-1], data["proprio"][:-1], data["previous_action"][:-1]))
            y.append(data["requested_action"].astype(np.float32))
    if not x:
        raise ValueError("Training requires at least one episode")
    x, y = np.concatenate(x), np.concatenate(y)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite training data")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.05)
    # Balance the image and proprio/action groups rather than counting pixels as
    # hundreds of independent votes against a few robot coordinates.
    scale[:192] *= np.sqrt(192)
    scale[192:] *= np.sqrt(x.shape[1] - 192)
    x = (x - mean) / scale
    np.savez_compressed(Path(output_dir) / "nearest-policy.npz", x=x, y=y, mean=mean, scale=scale)
    return NearestPolicy(x, y, mean, scale)


def load_nearest(checkpoint_dir):
    """Load the baseline without pickle or executable checkpoint deserialization."""
    with np.load(Path(checkpoint_dir) / "nearest-policy.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("x", "y", "mean", "scale")}
    x, y, mean, scale = (arrays[key] for key in ("x", "y", "mean", "scale"))
    if (
        x.ndim != 2
        or not len(x)
        or x.shape[1] != 222
        or y.shape != (len(x), 7)
        or mean.shape != (222,)
        or scale.shape != (222,)
        or any(not np.all(np.isfinite(value)) for value in arrays.values())
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid Panda visual baseline checkpoint")
    return NearestPolicy(x, y, mean, scale)
