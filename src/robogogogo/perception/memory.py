"""Bounded world-frame surface memory with conservative measured-ray invalidation.

Memory guides sensing, never authorizes contact. Missing depth is unknown, and
occluded surfaces age in place rather than following an unobserved object pose.
"""

from dataclasses import replace

import numpy as np

from .recovery import detect, foreground
from .view_quality import ViewEvidence


class SceneMemory:
    def __init__(self, *, capacity=4000, lifetime_s=8.0, process_noise_m=0.008):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Scene memory capacity must be a positive integer")
        if (
            not np.isfinite([lifetime_s, process_noise_m]).all()
            or min(lifetime_s, process_noise_m) <= 0
        ):
            raise ValueError("Scene memory lifetime and process noise must be positive")
        self.capacity, self.lifetime_s, self.process_noise_m = capacity, lifetime_s, process_noise_m
        self.cells = {}
        self.timestamp = None
        self.stats = {}

    def clear(self):
        self.cells.clear()
        self.timestamp = None
        self.stats = {}

    def _contradicted(self, points, cloud):
        """Erase only where a measured ray reaches beyond the remembered surface.

        A closer return wins over a farther one; invalid pixels and uncovered
        sight lines cannot erase memory. Ray tubes approximate the voxel footprint.
        """
        cleared = np.zeros(len(points), bool)
        for pose, calibration in cloud.views:
            center = np.asarray(pose["position"])
            if "depth" in calibration and "intrinsics" in calibration:
                depth = calibration["depth"]
                optical = (points - center) @ np.asarray(pose["rotation"])
                pixels = optical @ calibration["intrinsics"].T
                pixels = np.divide(
                    pixels[:, :2],
                    optical[:, 2, None],
                    out=np.zeros((len(points), 2)),
                    where=optical[:, 2, None] > 0,
                )
                uv = np.rint(pixels).astype(int)
                valid = (
                    (optical[:, 2] > 0)
                    & (uv[:, 0] >= 1)
                    & (uv[:, 0] < depth.shape[1] - 1)
                    & (uv[:, 1] >= 1)
                    & (uv[:, 1] < depth.shape[0] - 1)
                )
                indices = np.flatnonzero(valid)
                neighborhood = np.stack(
                    [
                        depth[uv[indices, 1] + dy, uv[indices, 0] + dx]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                    ],
                    axis=1,
                )
                # Require all neighboring pixels to see beyond the old point.
                # A hole or foreground edge is not free-space evidence.
                margin = 0.008 + 3 * cloud.uncertainty_m
                cleared[indices] |= np.isfinite(neighborhood).all(axis=1) & (
                    neighborhood > optical[indices, 2, None] + margin
                ).all(axis=1)
                continue
            if "surface_points" not in calibration:
                continue
            rays = calibration["surface_points"] - center
            length = np.linalg.norm(rays, axis=1)
            valid = length > 1e-6
            rays, length = rays[valid], length[valid]
            if not len(rays):
                continue
            # Uniform sampling bounds processing without claiming unknown rays free.
            indices = np.linspace(0, len(rays) - 1, min(4000, len(rays)), dtype=int)
            directions, length = rays[indices] / length[indices, None], length[indices]
            for start in range(0, len(points), 64):
                relative = points[start : start + 64] - center
                along = relative @ directions.T
                squared = np.sum(relative * relative, axis=1)[:, None] - along * along
                matches = (squared < 0.006**2) & (along > 0)
                margin = 0.008 + 3 * cloud.uncertainty_m
                farther = matches & (length > along + margin)
                nearer = matches & (length <= along + margin)
                cleared[start : start + 64] |= farther.any(axis=1) & ~nearer.any(axis=1)
        return cleared

    def observe(self, cloud, *, seed=None, target_mask=None):
        # Validate and detect on this frame only before changing persistent state.
        observation = detect(cloud, seed=seed, target_mask=target_mask)
        now = cloud.timestamp
        if self.timestamp is not None and now < self.timestamp:
            raise ValueError(
                "Scene memory rejects out-of-order observations; clear it after a clock reset"
            )
        mask = (
            (foreground(cloud, seed) if seed is not None else np.zeros(len(cloud.points), bool))
            if target_mask is None
            else np.asarray(target_mask, bool)
        )
        dt = 0 if self.timestamp is None else now - self.timestamp
        keys = list(self.cells)
        cleared = self._contradicted(
            np.array([self.cells[k]["point"] for k in keys]).reshape(-1, 3), cloud
        )
        expired = 0
        target_changes = sum(
            self.cells[k]["target"] for k, removed in zip(keys, cleared) if removed
        )
        for key, removed in zip(keys, cleared):
            cell = self.cells[key]
            if removed or now - cell["seen"] > self.lifetime_s:
                del self.cells[key]
                expired += not removed
            else:
                cell["variance"] += self.process_noise_m**2 * dt
                if target_changes and cell["target"]:
                    # Motion evidence makes still-occluded parts of the payload
                    # unreliable. Do not translate them with an invented rigid pose.
                    cell["variance"] += 0.02**2
        voxel = np.floor(cloud.points / 0.008).astype(int)
        _, selected = np.unique(voxel, axis=0, return_index=True)
        for index in selected:
            key = tuple(voxel[index])
            old = self.cells.get(key)
            variance = cloud.uncertainty_m**2
            point = cloud.points[index].copy()
            if old is not None and old["seen"] < now:
                weight = old["variance"] / (old["variance"] + variance)
                point = old["point"] * (1 - weight) + point * weight
                # Repeated views are correlated: don't claim unlimited certainty.
                variance = max(variance * 0.5, variance * weight)
                innovation = np.linalg.norm(cloud.points[index] - old["point"])
                if innovation > 3 * cloud.uncertainty_m:
                    # Even inside one voxel, inconsistent measurements must not
                    # make a moving surface look more certain or lag behind it.
                    point = cloud.points[index].copy()
                    variance = max(cloud.uncertainty_m**2, (innovation / 2) ** 2)
            self.cells[key] = dict(
                point=point,
                normal=cloud.normals[index].copy(),
                color=cloud.colors[index].copy(),
                target=bool(mask[index]),
                variance=variance,
                seen=now,
            )
        if len(self.cells) > self.capacity:
            current = [k for k in self.cells if self.cells[k]["seen"] == now]
            hidden = [k for k in self.cells if self.cells[k]["seen"] < now]
            hidden_budget = min(len(hidden), self.capacity // 4)
            current_budget = min(len(current), self.capacity - hidden_budget)
            hidden_budget = min(len(hidden), self.capacity - current_budget)
            retained = []
            for group, budget in ((current, current_budget), (hidden, hidden_budget)):
                # Preserve spatial coverage within each class instead of truncating
                # one end of the world; reserve space for occluded scene history.
                targets = [k for k in group if self.cells[k]["target"]]
                other = [k for k in group if not self.cells[k]["target"]]
                target_budget = min(len(targets), max(budget // 2, budget - len(other)))
                for keys_to_keep, count in (
                    (targets, target_budget),
                    (other, budget - target_budget),
                ):
                    if count:
                        retained.extend(
                            keys_to_keep[i]
                            for i in np.linspace(0, len(keys_to_keep) - 1, count, dtype=int)
                        )
            self.cells = {k: self.cells[k] for k in retained}
        self.timestamp = now
        cells = list(self.cells.values())
        points = np.array([c["point"] for c in cells]).reshape(-1, 3)
        normals = np.array([c["normal"] for c in cells]).reshape(-1, 3)
        target = np.array([c["target"] for c in cells], bool)
        uncertainty = np.sqrt([c["variance"] for c in cells])
        age = np.array([now - c["seen"] for c in cells])
        # No historical camera poses: old viewpoints must not count as current
        # coverage of moving objects. Keep only this capture's calibration.
        evidence = ViewEvidence(
            points,
            normals,
            target,
            cloud.uncertainty_m,
            cloud.views,
            np.exp(-age / 2) * np.minimum(1, cloud.uncertainty_m / np.maximum(uncertainty, 1e-9)),
        )
        self.stats = dict(
            surfaces=len(cells),
            remembered=int(np.sum(age > 0)),
            invalidated=int(cleared.sum()),
            expired=int(expired),
            motion_evidence=int(target_changes),
            max_uncertainty_m=float(uncertainty.max()) if len(cells) else 0.0,
        )
        diagnostics = dict(observation.diagnostics, scene_memory=dict(self.stats))
        return replace(observation, diagnostics=diagnostics, view_evidence=evidence)
