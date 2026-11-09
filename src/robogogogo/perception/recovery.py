"""Observed-surface proposals for recovery. No simulator model or object state input.

Unknown pixels stay unknown. Geometry proposals need subsequent kinematic,
collision, contact-force and stability verification; they are not grasp guarantees.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SurfaceCloud:
    points: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    timestamp: float
    uncertainty_m: float
    source: str


@dataclass(frozen=True)
class RecoveryObservation:
    timestamp: float
    source: str
    uncertainty_m: float
    grasps: tuple
    supports: tuple
    object_axis: np.ndarray | None
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        return dict(
            time=self.timestamp,
            source=self.source,
            uncertainty_m=self.uncertainty_m,
            grasp_count=len(self.grasps),
            support_count=len(self.supports),
            diagnostics=self.diagnostics,
        )


def backproject(
    depth, rgb, intrinsics, position, rotation, *, timestamp, uncertainty_m=0.005, source="rgbd"
):
    """Metric optical-Z depth and optical camera-to-world calibration -> visible cloud."""
    depth, rgb, k, r, c = map(np.asarray, (depth, rgb, intrinsics, rotation, position))
    if (
        depth.ndim != 2
        or min(depth.shape) < 3
        or rgb.shape != (*depth.shape, 3)
        or k.shape != (3, 3)
        or r.shape != (3, 3)
        or c.shape != (3,)
        or not all(np.isfinite(a).all() for a in (k, r, c))
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(r), 1, atol=1e-5)
        or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-10
        or abs(k[1, 0]) > 1e-10
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.isfinite(timestamp)
        or not 0 < uncertainty_m <= 0.02
    ):
        raise ValueError("Invalid calibrated depth observation")
    yy, xx = np.indices(depth.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(k).T
    points = (rays * depth[..., None]) @ r.T + c
    valid = np.isfinite(depth) & (depth > 0)
    dx, dy = np.zeros_like(points), np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    good = (
        valid
        & (lengths > 1e-10)
        & (np.linalg.norm(dx, axis=-1) < 0.04)
        & (np.linalg.norm(dy, axis=-1) < 0.04)
    )
    normals = np.divide(
        normals, lengths[..., None], out=np.zeros_like(normals), where=good[..., None]
    )
    return SurfaceCloud(
        points[valid], normals[valid], rgb[valid], float(timestamp), float(uncertainty_m), source
    )


def foreground(cloud, seed):
    """Classical color/3D region growing from a visible point near a proprioceptive seed.

    Optional learned segmentation can instead supply a point mask to detect().
    A seed alone never supplies object dimensions or predefined grasp sites.
    """
    points, colors = cloud.points, cloud.colors.astype(float)
    seed = np.asarray(seed, dtype=float)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise ValueError("Target seed must be a finite world point")
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    distances = np.linalg.norm(points - seed, axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] > 0.05:
        return np.zeros(len(points), dtype=bool)
    eligible = (np.linalg.norm(colors - colors[nearest], axis=1) < 35) & (distances < 0.35)
    # Voxel connectivity prevents merging a similarly colored remote surface.
    keys = np.floor(points / 0.012).astype(int)
    cells = {tuple(k) for k in keys[eligible]}
    connected, pending = set(), [tuple(keys[nearest])]
    while pending:
        key = pending.pop()
        if key in connected or key not in cells:
            continue
        connected.add(key)
        pending.extend(
            tuple(key[j] + (d if j == axis else 0) for j in range(3))
            for axis in range(3)
            for d in (-1, 1)
        )
    return eligible & np.array([tuple(k) in connected for k in keys])


def detect(cloud, *, seed=None, target_mask=None, max_grasps=24, max_supports=24):
    """Derive opposing-surface grasp sites and fully observed horizontal landing patches.

    Masks come from image segmentation, never simulator body IDs. Supports use
    occupied planar grid cells; holes/occlusions cannot become landing area.
    """
    p, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    if p.ndim != 2 or p.shape[1:] != (3,) or normals.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Invalid surface cloud")
    if target_mask is None:
        target_mask = foreground(cloud, seed) if seed is not None else np.zeros(len(p), bool)
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (len(p),):
        raise ValueError("Target mask must match the visible point cloud")
    obj = p[mask]
    grasps, supports, axis = [], [], None
    uncertainty = cloud.uncertainty_m
    if len(obj) >= 40:
        center = np.median(obj, axis=0)
        _, _, vh = np.linalg.svd(obj - center, full_matrices=False)
        axis = vh[0]
        coordinate = (obj - center) @ axis
        # Adaptive slices follow observed geometry rather than model cell indices.
        for t in np.arange(coordinate.min() + 0.015, coordinate.max() - 0.015, 0.015):
            selected = np.abs(coordinate - t) < 0.018
            section, n = obj[selected], normals[mask][selected]
            if len(section) < 12:
                continue
            for closing in vh[1:]:
                lateral = section @ closing
                lo, hi = np.quantile(lateral, [0.05, 0.95])
                # Both visible opposing flanks, with normals parallel to jaw closure.
                side = np.abs(n @ closing) > 0.7
                left_mask = side & (lateral < lo + 0.006)
                right_mask = side & (lateral > hi - 0.006)
                left, right = section[left_mask], section[right_mask]
                width = hi - lo + 2 * uncertainty
                if len(left) < 3 or len(right) < 3 or not 0.012 < width < 0.085:
                    continue
                # Depth normals have consistent camera-ray orientation. Same-facing
                # edge patches are not evidence of two opposing grasp flanks.
                if np.median(n[left_mask] @ closing) * np.median(n[right_mask] @ closing) > -0.49:
                    continue
                point = (np.median(left, axis=0) + np.median(right, axis=0)) / 2
                grasps.append(
                    dict(
                        position=point.tolist(),
                        closing_axis=closing.tolist(),
                        width_m=float(width),
                        evidence_points=int(len(left) + len(right)),
                    )
                )
    # Height bins fit near-horizontal surfaces. Reject slope and mixed-height cells.
    planar = (~mask) & (np.abs(normals[:, 2]) > 0.985)
    planar_points = p[planar]
    resolution = 0.01
    if len(planar_points):
        bins = np.round(planar_points[:, 2] / 0.008).astype(int)
        height_order = sorted(
            set(bins),
            key=lambda value: (
                abs(value * 0.008 - np.asarray(seed)[2]) if seed is not None else -value
            ),
        )
        for height_bin in height_order:
            surface = planar_points[np.abs(planar_points[:, 2] - height_bin * 0.008) < 0.006]
            if len(surface) < 36:
                continue
            xy = np.floor(surface[:, :2] / resolution).astype(int)
            occupied = {tuple(key) for key in xy}
            # Use the observed object footprint, otherwise a conservative 8 cm patch.
            footprint = np.ptp(obj[:, :2], axis=0) if len(obj) >= 40 else np.array([0.08, 0.08])
            half = np.maximum(0.03, footprint / 2 + 0.01 + 2 * uncertainty)
            radius = np.ceil(half / resolution).astype(int)
            cells = sorted(
                occupied,
                key=lambda key: (
                    np.linalg.norm((np.asarray(key) + 0.5) * resolution - np.asarray(seed)[:2])
                    if seed is not None
                    else key
                ),
            )
            for key in cells:
                if len(supports) >= max_supports:
                    break
                if any(
                    np.linalg.norm(np.array(s["position"][:2]) - (np.array(key) + 0.5) * resolution)
                    < 0.04
                    for s in supports
                ):
                    continue
                patch = {
                    (key[0] + dx, key[1] + dy)
                    for dx in range(-radius[0], radius[0] + 1)
                    for dy in range(-radius[1], radius[1] + 1)
                }
                if not patch.issubset(occupied):
                    continue
                local = surface[np.array([tuple(k) in patch for k in xy])]
                if np.ptp(local[:, 2]) > 2 * uncertainty + 0.003:
                    continue
                patch_center = (np.array(key) + 0.5) * resolution
                occupied_above = (
                    ~mask
                    & (np.abs(p[:, :2] - patch_center) <= half).all(axis=1)
                    & (p[:, 2] > np.median(local[:, 2]) + 2 * uncertainty + 0.005)
                    & (p[:, 2] < np.median(local[:, 2]) + 0.25)
                )
                if occupied_above.any():
                    continue
                position = [*((np.array(key) + 0.5) * resolution), float(np.median(local[:, 2]))]
                supports.append(
                    dict(position=position, half_size=half.tolist(), evidence_points=len(local))
                )
    return RecoveryObservation(
        cloud.timestamp,
        cloud.source,
        uncertainty,
        tuple(
            grasps[i]
            for i in np.linspace(
                0, max(0, len(grasps) - 1), min(len(grasps), max_grasps), dtype=int
            )
        ),
        tuple(supports),
        axis,
        dict(visible_points=len(p), target_points=len(obj), unknown_landing_cells_rejected=True),
    )


def fuse(clouds):
    """Fuse synchronized calibrated visible surfaces, preserving missing regions."""
    clouds = tuple(clouds)
    if not clouds or max(c.timestamp for c in clouds) - min(c.timestamp for c in clouds) > 0.03:
        raise ValueError("Recovery views must be synchronized within 30 ms")
    return SurfaceCloud(
        np.concatenate([c.points for c in clouds]),
        np.concatenate([c.normals for c in clouds]),
        np.concatenate([c.colors for c in clouds]),
        min(c.timestamp for c in clouds),
        max(c.uncertainty_m for c in clouds),
        "+".join(sorted({c.source for c in clouds})),
    )
