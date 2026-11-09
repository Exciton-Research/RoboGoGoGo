"""Next-view utility from measured surfaces, never simulator collision geometry.

Unknown boundary probes are exploration hypotheses, not detected grasp/support
sites. Scores predict useful measurements; only a new image can confirm them.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewEvidence:
    points: np.ndarray
    normals: np.ndarray
    target: np.ndarray
    uncertainty_m: float
    views: tuple = ()

    @classmethod
    def from_cloud(cls, cloud, mask):
        # One measured sample per 8 mm cell, bounded independently of image size.
        _, indices = np.unique(np.floor(cloud.points / 0.008), axis=0, return_index=True)
        indices = indices[np.linspace(0, len(indices) - 1, min(4000, len(indices)), dtype=int)]
        return cls(
            cloud.points[indices].copy(),
            cloud.normals[indices].copy(),
            mask[indices].copy(),
            float(cloud.uncertainty_m),
            cloud.views,
        )

    def probes(self, support):
        selected = ~self.target & (np.abs(self.normals[:, 2]) > 0.985) if support else self.target
        points, normals = self.points[selected], self.normals[selected]
        if not len(points):
            # A scene with no recognized target can still expose boundaries of
            # observed occluders. It cannot supply a fabricated target location.
            points, normals = self.points, self.normals
        indices = np.linspace(0, len(points) - 1, min(160, len(points)), dtype=int)
        points, normals = points[indices], normals[indices]
        offsets = (
            np.array([[0.04, 0, 0], [-0.04, 0, 0], [0, 0.04, 0], [0, -0.04, 0]])
            if support
            else np.array([[0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0, -0.02, 0]])
        )
        probes = np.concatenate([points] + [points + offset for offset in offsets])
        return probes, np.tile(normals, (5, 1)), len(points)

    def quality(self, pose, camera, probes, normals):
        center, rotation = np.asarray(pose["position"]), np.asarray(pose["rotation"])
        rays = probes - center
        distance = np.linalg.norm(rays, axis=1)
        optical = rays @ rotation
        tangent = np.tan(np.deg2rad(camera.get("vertical_fov_degrees", 70) / 2))
        aspect = camera.get("width", 320) / camera.get("height", 240)
        visible = (
            (optical[:, 2] > 0.08)
            & (np.abs(optical[:, 0]) < optical[:, 2] * tangent * aspect)
            & (np.abs(optical[:, 1]) < optical[:, 2] * tangent)
        )
        directions = rays / np.maximum(distance[:, None], 1e-9)
        blockers = self.points - center
        # Ray/surfel tests in small batches: surfaces can occlude, missing space
        # remains unknown. A tolerance avoids treating a probe's own surface as
        # an occluder. No collision-world/hidden-object query is made here.
        for start in range(0, len(probes), 64):
            end = start + 64
            along = blockers @ directions[start:end].T
            squared = np.sum(blockers * blockers, axis=1)[:, None] - along * along
            blocked = (
                (along > 0) & (along < distance[start:end] - 0.016) & (squared < 0.009**2)
            ).any(axis=0)
            visible[start:end] &= ~blocked
        incidence = np.abs(np.sum(normals * directions, axis=1))
        incidence = np.where(np.linalg.norm(normals, axis=1) > 0.5, incidence, 0.25)
        # Conservative range/incidence noise model; not a calibrated posterior.
        sigma = self.uncertainty_m * (distance / 0.5) ** 2 / np.maximum(incidence, 0.2)
        quality = visible / (1 + (sigma / 0.005) ** 2)
        return quality, visible

    def baseline(self, before, camera, *, support=False):
        probes, normals, measured = self.probes(support)
        old, old_visible = self.quality(before, camera, probes, normals)
        for pose, calibration in self.views:
            observed, observed_visible = self.quality(pose, calibration, probes, normals)
            old = np.maximum(old, observed)
            old_visible |= observed_visible
        return probes, normals, measured, old, old_visible

    def score(self, before, after, camera, *, support=False, baseline=None):
        probes, normals, measured, old, old_visible = (
            self.baseline(before, camera, support=support) if baseline is None else baseline
        )
        if not measured:
            return dict(gain=0.0, newly_visible=0, visible=0, probes=0)
        new, visible = self.quality(after, camera, probes, normals)
        # Synchronized reference views already cover some regions. Reward only
        # incremental quality beyond their best measurement; unknown probes are discounted.
        weights = np.ones(len(probes))
        weights[measured:] = 0.35
        gain = float(np.sum(weights * np.maximum(new - old, 0)) / weights.sum())
        return dict(
            gain=gain,
            newly_visible=int(np.sum(visible & ~old_visible)),
            visible=int(visible.sum()),
            probes=len(probes),
        )
