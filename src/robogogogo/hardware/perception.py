"""Read-only calibrated hardware perception worker, separate from simulation state."""

import copy
import hashlib
import json
import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..perception import SceneMemory, SurfaceCloud, backproject, fuse
from ..perception.stereo import estimate_stereo, opencv
from .synchronization import Synchronizer


class HardwarePerception:
    def __init__(self, profile, *, audit_dir=None, clock=time.monotonic):
        self.profile, self.clock = profile, clock
        self.sync = Synchronizer(profile)
        self.memory = SceneMemory()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.updated = None
        self.count = 0
        self.verified_info = set()
        self.status = dict(state="waiting", reason=self.sync.reason, scene_memory={})
        self.scene = None
        self.audit = None
        if audit_dir is not None:
            path = Path(audit_dir) / str(uuid4())
            path.mkdir(parents=True, exist_ok=False)
            (path / "calibration.json").write_text(json.dumps(profile.document, indent=2))
            (path / "robot.urdf").write_text(profile.robot_xml)
            self.audit = RotatingFileHandler(
                path / "observations.jsonl", maxBytes=10_000_000, backupCount=2
            )
            self.audit.setFormatter(logging.Formatter("%(message)s"))

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="hardware-perception")
        self.thread.start()

    def ingest(self, sample):
        with self.lock:
            self.sync.add(sample, self.clock())

    def calibration_info(self, info):
        with self.lock:
            camera = self.profile.cameras.get(info.name)
            if camera is None:
                return
            try:
                if (
                    info.frame_id != camera["frame_id"]
                    or (info.width, info.height) != (camera["width"], camera["height"])
                    or info.distortion_model != camera["distortion_model"]
                    or np.asarray(info.intrinsics).shape != (3, 3)
                    or np.asarray(info.distortion).shape != np.asarray(camera["distortion"]).shape
                    or not np.allclose(info.intrinsics, camera["intrinsics"], rtol=1e-6, atol=1e-6)
                    or not np.allclose(info.distortion, camera["distortion"], rtol=1e-6, atol=1e-6)
                ):
                    raise ValueError("Driver CameraInfo differs from measured profile")
                self.verified_info.add(info.name)
            except (ValueError, TypeError) as error:
                self.sync.fault = f"Calibration mismatch for {info.name}: {error}"

    def invalidate(self, reason):
        with self.lock:
            self.sync.fault = str(reason)

    def scene_snapshot(self):
        with self.lock:
            if (
                self.scene is None
                or self.sync.fault
                or self.updated is None
                or self.clock() - self.updated > 0.5
                or self.status["state"] == "fault"
            ):
                raise ValueError("No fresh calibrated hardware scene is available")
            return copy.deepcopy(self.scene)

    def snapshot(self):
        with self.lock:
            result = dict(self.status)
            result.update(
                calibration_revision=self.profile.document["revision"],
                calibration_digest=self.profile.digest,
                world_frame=self.profile.document["world_frame"],
                clock_domain=self.profile.document["clock_domain"],
                bundles=self.count,
                dropped=self.sync.dropped,
                rejected=self.sync.rejected,
                last_rejection=self.sync.last_rejection,
                cameras=[
                    dict(
                        name=n,
                        frame_id=c["frame_id"],
                        mount_link=c["mount_link"],
                        width=c["width"],
                        height=c["height"],
                        reprojection_error_px=c["reprojection_error_px"],
                    )
                    for n, c in self.profile.cameras.items()
                ],
            )
            if self.sync.fault:
                result.update(state="fault", reason=self.sync.fault, scene_memory={})
            if self.updated is not None:
                result["age_s"] = max(0, self.clock() - self.updated)
                if result["age_s"] > 0.5 and not self.sync.fault:
                    result["state"] = "stale"
            return result

    def _run(self):
        while not self.stop.wait(0.05):
            try:
                self.step()
            except Exception as error:
                with self.lock:
                    self.status.update(state="fault", reason=str(error))
                    self.scene = None
                return

    def step(self):
        with self.lock:
            required_info = {
                n for n, c in self.profile.cameras.items() if c.get("require_camera_info", False)
            }
            if not self.sync.fault and not required_info.issubset(self.verified_info):
                self.status.update(
                    state="blocked",
                    reason="Waiting for matching CameraInfo: "
                    + ", ".join(sorted(required_info - self.verified_info)),
                )
                return
            bundle = self.sync.next(self.clock())
            if bundle is None:
                self.status.update(
                    state="blocked" if self.sync.fault else "waiting", reason=self.sync.reason
                )
                if self.sync.fault:
                    self.memory.clear()
                    self.status["scene_memory"] = {}
                    self.scene = None
                return
        cv = opencv()
        images, poses = {}, {}
        for name, frame in bundle.frames.items():
            camera = self.profile.cameras[name]
            decoded = cv.imdecode(np.frombuffer(frame.data, np.uint8), cv.IMREAD_COLOR)
            if decoded is None or decoded.shape != (camera["height"], camera["width"], 3):
                raise ValueError(f"Decoded image dimensions differ from calibration: {name}")
            images[name] = cv.cvtColor(
                cv.undistort(
                    decoded, np.asarray(camera["intrinsics"]), np.asarray(camera["distortion"])
                ),
                cv.COLOR_BGR2RGB,
            )
            poses[name] = self.profile.pose(name, bundle.positions[name])
        clouds, rejected = [], []
        for left, right in self.profile.pairs:
            try:
                result = estimate_stereo(
                    images[left],
                    images[right],
                    self.profile.cameras[left],
                    self.profile.cameras[right],
                    poses[left],
                    poses[right],
                    num_disparities=min(64, (images[left].shape[1] // 32) * 16),
                )
                pose_error = max(
                    self.profile.pose_uncertainty(
                        n, bundle.positions[n], bundle.joint_gap_s, bundle.skew_s
                    )
                    for n in (left, right)
                )
                sigma = result.depth**2 / (result.focal_px * result.baseline_m) + pose_error
                valid = np.isfinite(result.depth) & (sigma <= 0.02)
                if not valid.any():
                    raise ValueError("No reliable metric depth within the 20 mm uncertainty budget")
                clouds.append(
                    backproject(
                        np.where(valid, result.depth, np.nan),
                        result.rectified_left,
                        result.intrinsics,
                        result.position,
                        result.rotation,
                        timestamp=bundle.timestamp,
                        uncertainty_m=max(0.003, float(sigma[valid].max())),
                        source="hardware_calibrated_stereo",
                    )
                )
            except ValueError as error:
                rejected.append(dict(pair=[left, right], reason=str(error)))
        if clouds:
            observation = self.memory.observe(fuse(clouds))
        else:
            empty = np.empty((0, 3))
            observation = self.memory.observe(
                SurfaceCloud(empty, empty, empty, bundle.timestamp, 0.02, "hardware_no_depth")
            )
        record = dict(
            timestamp=bundle.timestamp,
            calibration_digest=bundle.calibration_digest,
            skew_s=bundle.skew_s,
            joint_gap_s=bundle.joint_gap_s,
            frames={
                n: dict(
                    timestamp=f.timestamp,
                    frame_id=f.frame_id,
                    sha256=hashlib.sha256(f.data).hexdigest(),
                    joints=bundle.positions[n],
                    optical_to_world={k: v.tolist() for k, v in poses[n].items()},
                )
                for n, f in bundle.frames.items()
            },
            rejected_pairs=rejected,
            scene_memory=observation.diagnostics["scene_memory"],
        )
        if self.audit:
            self.audit.emit(
                logging.makeLogRecord(
                    dict(msg=json.dumps(record, allow_nan=False), levelno=logging.INFO)
                )
            )
        with self.lock:
            if self.sync.fault or self.stop.is_set():
                self.memory.clear()
                return
            self.scene = dict(
                timestamp=bundle.timestamp,
                calibration_digest=self.profile.digest,
                world_frame=self.profile.document["world_frame"],
                points=observation.view_evidence.points.tolist(),
                normals=observation.view_evidence.normals.tolist(),
                confidence=observation.view_evidence.confidence.tolist(),
                last_seen=[c["seen"] for c in self.memory.cells.values()],
                uncertainty_m=[float(np.sqrt(c["variance"])) for c in self.memory.cells.values()],
            )
            self.count += 1
            self.updated = self.clock()
            self.status = dict(
                state="observing" if clouds else "blocked",
                reason="Calibrated depth fused into hardware scene memory"
                if clouds
                else "No stereo pair produced reliable depth",
                **record,
            )

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise ValueError("Hardware perception worker did not stop")
        if self.audit:
            self.audit.close()
        self.memory.clear()
