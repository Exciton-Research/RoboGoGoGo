"""Bounded, explicit ROS TF chains with exposure-time SE(3) interpolation.

This consumes a localization system's estimates; it does not run SLAM or infer
missing transforms. No latest-transform substitution or extrapolation is allowed.
"""

from dataclasses import dataclass

import numpy as np


def quaternion(value):
    q = np.asarray(value, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.001:
        raise ValueError("Localization quaternion must be finite and normalized (x,y,z,w)")
    return q / np.linalg.norm(q)


def angular_distance(a, b):
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


def slerp(a, b, fraction):
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + fraction * (b - a)
        return q / np.linalg.norm(q)
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


def matrix(position, q):
    x, y, z, w = q
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = position
    return result


@dataclass(frozen=True)
class BaseEstimate:
    world_from_base: np.ndarray
    gap_s: float
    epoch: int
    translation_uncertainty_m: float
    rotation_uncertainty_rad: float
    interpolation_translation_m: float
    interpolation_rotation_rad: float
    edges: tuple


class LocalizationBuffer:
    def __init__(self, config, clock_domain):
        self.config, self.clock_domain = config, clock_domain
        self.chain = tuple(config["tf_chain"])
        self.edges = tuple(zip(self.chain, self.chain[1:]))
        self.dynamic = {edge: [] for edge in self.edges}
        self.static = {}
        self.highwater = {}
        self.source_epochs = {}
        self.epoch = 0
        self.resets = 0
        self.last_reset = None
        self.fault = None

    def _reset(self, reason):
        self.epoch += 1
        self.resets += 1
        self.last_reset = reason
        for queue in self.dynamic.values():
            queue.clear()

    def add(self, sample, received):
        if self.fault:
            return
        edge = (sample.parent_frame, sample.child_frame)
        if sample.child_frame not in self.chain[1:]:
            return  # Other robot TF edges belong to neither localization nor this chain.
        try:
            if edge not in self.dynamic:
                raise ValueError(
                    "Localization TF parent changed or conflicts with configured chain"
                )
            if sample.clock_domain != self.clock_domain:
                raise ValueError("Localization clock domain differs from camera/joint clock")
            position = np.asarray(sample.translation, float)
            q = quaternion(sample.quaternion_xyzw)
            if (
                position.shape != (3,)
                or not np.isfinite(position).all()
                or not np.isfinite([sample.timestamp, received]).all()
                or sample.timestamp < 0
                or type(sample.static) is not bool
            ):
                raise ValueError("Invalid localization timestamp or translation")
            if sample.epoch != self.source_epochs.get(edge, sample.epoch):
                self._reset("Localization provider changed its epoch")
            self.source_epochs[edge] = sample.epoch
            item = (sample.timestamp, position, q, received)
            if sample.static:
                if self.dynamic[edge] or edge in self.highwater:
                    raise ValueError("Localization edge changed from dynamic to static")
                old = self.static.get(edge)
                if old and (
                    np.linalg.norm(old[1] - position) > 1e-9 or angular_distance(old[2], q) > 1e-7
                ):
                    self._reset("Static localization transform changed")
                self.static[edge] = item
                if len(self.static) == len(self.edges):
                    raise ValueError("Moving-base localization requires a dynamic TF edge")
                return
            if edge in self.static:
                raise ValueError("Localization edge changed from static to dynamic")
            if sample.timestamp < self.highwater.get(edge, sample.timestamp) - 1:
                raise ValueError("Localization clock reset; reconnect required")
            queue = self.dynamic[edge]
            for old in queue:
                if old[0] == sample.timestamp:
                    if (
                        np.linalg.norm(old[1] - position) > 1e-9
                        or angular_distance(old[2], q) > 1e-7
                    ):
                        raise ValueError(
                            "Conflicting localization transforms at the same timestamp"
                        )
                    return
            # Any newly inserted edge must be continuous with both neighbors;
            # an out-of-order discontinuity is ambiguous and cannot reset history safely.
            before = [item for item in queue if item[0] < sample.timestamp]
            after = [item for item in queue if item[0] > sample.timestamp]
            for old in before[-1:] + after[:1]:
                dt = abs(sample.timestamp - old[0])
                distance = np.linalg.norm(position - old[1])
                angle = angular_distance(q, old[2])
                jump = (
                    distance > self.config["jump_translation_m"]
                    or angle > self.config["jump_rotation_rad"]
                    or distance > self.config["max_linear_speed_m_s"] * dt + 1e-6
                    or angle > self.config["max_angular_speed_rad_s"] * dt + 1e-6
                )
                if jump:
                    if after:
                        raise ValueError(
                            "Out-of-order localization discontinuity; reconnect required"
                        )
                    self._reset("Localization pose jumped or exceeded configured motion bounds")
                    break
            queue.append(item)
            queue.sort(key=lambda value: value[0])
            if len(queue) > 256:
                del queue[:-256]
            self.highwater[edge] = max(sample.timestamp, self.highwater.get(edge, sample.timestamp))
        except (ValueError, TypeError) as error:
            self.fault = str(error)

    def health(self, now):
        if self.fault:
            return self.fault
        for edge, queue in self.dynamic.items():
            if edge in self.static:
                continue
            if not queue:
                return f"Waiting for localization TF: {edge[0]} → {edge[1]}"
            if now - queue[-1][3] > self.config["max_age_s"]:
                return f"Stale localization TF: {edge[0]} → {edge[1]}"
        return None

    def at(self, timestamp, now, time_uncertainty_s=0.0):
        problem = self.health(now)
        if problem:
            raise ValueError(problem)
        composed, max_gap, provenance = np.eye(4), 0.0, []
        for edge in self.edges:
            if edge in self.static:
                a = b = self.static[edge]
            else:
                queue = self.dynamic[edge]
                before = [item for item in queue if item[0] <= timestamp]
                after = [item for item in queue if item[0] >= timestamp]
                if not before or not after:
                    raise ValueError(
                        "Waiting for TF bracketing exposure; localization extrapolation disabled"
                    )
                a, b = before[-1], after[0]
                if max(now - a[3], now - b[3]) > self.config["max_age_s"]:
                    raise ValueError("Localization bracket samples are stale")
            gap = b[0] - a[0]
            if gap > self.config["max_gap_s"]:
                raise ValueError("Localization TF interpolation gap exceeds limit")
            fraction = (timestamp - a[0]) / gap if gap else 0.0
            composed = composed @ matrix(
                a[1] + fraction * (b[1] - a[1]), slerp(a[2], b[2], fraction)
            )
            max_gap = max(max_gap, gap)
            provenance.append(
                dict(
                    parent=edge[0],
                    child=edge[1],
                    before=a[0],
                    after=b[0],
                    static=edge in self.static,
                )
            )
        # Speed bounds describe the composed world-to-base trajectory; sums of
        # bracket gaps conservatively account for independently timed chain edges.
        total_gap = sum(item["after"] - item["before"] for item in provenance)
        return BaseEstimate(
            composed,
            max_gap,
            self.epoch,
            self.config["translation_uncertainty_m"],
            self.config["rotation_uncertainty_rad"],
            self.config["max_linear_speed_m_s"] * (total_gap / 2 + time_uncertainty_s),
            self.config["max_angular_speed_rad_s"] * (total_gap / 2 + time_uncertainty_s),
            tuple(provenance),
        )
