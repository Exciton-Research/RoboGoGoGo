"""Bounded acquisition-time synchronization; never substitutes arrival timestamps."""

from dataclasses import dataclass, field

import numpy as np

from .contracts import CameraSample, JointSample, TransformSample
from .localization import LocalizationBuffer


@dataclass(frozen=True)
class SensorBundle:
    timestamp: float
    frames: dict
    positions: dict
    skew_s: float
    joint_gap_s: float
    calibration_digest: str
    base_poses: dict = field(default_factory=dict)
    localization_epoch: int = 0


class Synchronizer:
    def __init__(self, profile):
        self.profile = profile
        self.joints = []
        self.frames = {name: [] for name in profile.cameras}
        self.used = {name: -np.inf for name in profile.cameras}
        self.last_bundle = -np.inf
        self.rejected = 0
        self.dropped = 0
        self.reason = "Waiting for calibrated camera frames and bracketing joints"
        self.fault = None
        self.highwater = {}
        self.last_rejection = None
        self.localization = (
            LocalizationBuffer(profile.localization, profile.document["clock_domain"])
            if profile.localization
            else None
        )
        self.localization_epoch = 0

    def add(self, sample, received):
        if self.fault:
            return
        if isinstance(sample, TransformSample):
            if self.localization:
                self.localization.add(sample, received)
                if self.localization.epoch != self.localization_epoch:
                    self.localization_epoch = self.localization.epoch
                    self.joints.clear()
                    for queue in self.frames.values():
                        queue.clear()
                if self.localization.fault:
                    self.fault = self.localization.fault
            return
        if not np.isfinite([sample.timestamp, received]).all() or sample.timestamp < 0:
            self.rejected += 1
            self.last_rejection = "Invalid sensor timestamp"
            return
        if sample.clock_domain != self.profile.document["clock_domain"]:
            self.rejected += 1
            self.reason = "Clock domain missing or incompatible with calibration"
            self.last_rejection = self.reason
            return
        if isinstance(sample, JointSample):
            if not set(self.profile.tree.names).issubset(sample.names):
                self.reason = "Required robot joints missing"
                self.last_rejection = self.reason
                self.rejected += 1
                return
            queue, time = self.joints, sample.timestamp
            limit = 256
        elif isinstance(sample, CameraSample) and sample.name in self.frames:
            camera = self.profile.cameras[sample.name]
            if sample.frame_id != camera["frame_id"]:
                self.rejected += 1
                self.reason = f"Optical frame mismatch: {sample.name}"
                self.last_rejection = self.reason
                return
            queue, time = self.frames[sample.name], sample.timestamp + camera["time_offset_s"]
            limit = 16
        else:
            return
        stream = "joints" if isinstance(sample, JointSample) else "camera:" + sample.name
        if time < self.highwater.get(stream, time) - 1:
            self.fault = "Acquisition clock moved backward; reconnect to reset memory"
            self.reason = self.fault
            return
        self.highwater[stream] = max(time, self.highwater.get(stream, time))
        if isinstance(sample, CameraSample) and time <= self.used[sample.name]:
            return
        if any(item[0] == time for item in queue):
            return
        queue.append((time, sample, received))
        queue.sort(key=lambda item: item[0])
        if len(queue) > limit:
            del queue[0]
            self.dropped += 1
        # Image bytes are bounded globally as well as by stream count.
        while sum(len(item[1].data) for q in self.frames.values() for item in q) > 64 * 1024 * 1024:
            name = min((n for n, q in self.frames.items() if q), key=lambda n: self.frames[n][0][2])
            del self.frames[name][0]
            self.dropped += 1

    def interpolate(self, timestamp):
        before = [item for item in self.joints if item[0] <= timestamp]
        after = [item for item in self.joints if item[0] >= timestamp]
        if not before or not after:
            raise ValueError(
                "Waiting for joint samples bracketing exposure; extrapolation disabled"
            )
        a, b = before[-1], after[0]
        gap = b[0] - a[0]
        if gap > self.profile.max_gap:
            raise ValueError("Joint interpolation gap exceeds configured limit")
        qa, qb = dict(zip(a[1].names, a[1].position)), dict(zip(b[1].names, b[1].position))
        positions = {}
        for name in self.profile.tree.names:
            delta = qb[name] - qa[name]
            if self.profile.tree.joints[name]["kind"] == "continuous":
                if gap * self.profile.speed_limits[name] >= np.pi:
                    raise ValueError("Continuous-joint interpolation is ambiguous")
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
            if abs(delta) > self.profile.speed_limits[name] * gap + 1e-6:
                raise ValueError("Joint history violates configured velocity limits")
            positions[name] = qa[name] + (delta * (timestamp - a[0]) / gap if gap else 0)
        return positions, gap

    def next(self, now):
        if self.fault:
            self.reason = self.fault
            return None
        for queue in [self.joints, *self.frames.values()]:
            old = len(queue)
            queue[:] = [item for item in queue if now - item[2] <= 0.5]
            self.dropped += old - len(queue)
        if not self.joints or any(not queue for queue in self.frames.values()):
            self.reason = "Waiting for fresh frames from every configured camera and joint history"
            return None
        anchor = next(iter(self.frames))
        for reference in reversed(self.frames[anchor]):
            selected = {
                name: min(queue, key=lambda item: abs(item[0] - reference[0]))
                for name, queue in self.frames.items()
            }
            times = [item[0] for item in selected.values()]
            uncertainty = max(c["time_uncertainty_s"] for c in self.profile.cameras.values())
            skew = max(times) - min(times)
            if skew + 2 * uncertainty > self.profile.max_skew or max(times) <= self.last_bundle:
                self.reason = "Camera acquisition skew exceeds calibration tolerance"
                continue
            if self.joints[-1][0] - min(times) > 0.25:
                self.reason = "Camera acquisition is too old relative to robot state"
                continue
            try:
                estimates = {name: self.interpolate(item[0]) for name, item in selected.items()}
                bases = (
                    {
                        name: self.localization.at(
                            item[0], now, self.profile.cameras[name]["time_uncertainty_s"]
                        )
                        for name, item in selected.items()
                    }
                    if self.localization
                    else {}
                )
            except ValueError as error:
                self.reason = str(error)
                continue
            for name, item in selected.items():
                self.used[name] = item[0]
                self.frames[name][:] = [entry for entry in self.frames[name] if entry[0] > item[0]]
            self.last_bundle = max(times)
            self.reason = "Synchronized acquisition times with interpolated joints"
            return SensorBundle(
                max(times),
                {n: v[1] for n, v in selected.items()},
                {n: v[0] for n, v in estimates.items()},
                skew,
                max(v[1] for v in estimates.values()),
                self.profile.digest,
                base_poses=bases,
                localization_epoch=self.localization_epoch,
            )
        return None
