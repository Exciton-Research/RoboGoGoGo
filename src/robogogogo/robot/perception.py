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

from robogogogo.perception import SceneMemory, SurfaceCloud, backproject, fuse
from robogogogo.perception.stereo import estimate_stereo, opencv
from robogogogo.robot.synchronization import Synchronizer


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
        self.memory_epoch = 0
        self.audit = None
        if audit_dir is not None:
            path = Path(audit_dir) / str(uuid4())
            path.mkdir(parents=True, exist_ok=False)
            (path / "calibration.json").write_text(json.dumps(profile.document, indent=2))
            (path / "robot.urdf").write_text(profile.robot_xml)
            (path / "manifest.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        calibration_digest=profile.digest,
                        calibration_file="calibration.json",
                        robot_file="robot.urdf",
                    ),
                    indent=2,
                )
            )
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
            if self.sync.localization_epoch != self.memory_epoch or self.sync.fault:
                self.scene = None

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

    def localization_status(self):
        localization = self.sync.localization
        if localization is None:
            return dict(mode="fixed", epoch=0)
        return dict(
            mode="moving",
            epoch=localization.epoch,
            resets=localization.resets,
            chain=list(localization.chain),
            reason=localization.health(self.clock()),
            last_reset=localization.last_reset,
            translation_uncertainty_m=localization.config["translation_uncertainty_m"],
            rotation_uncertainty_rad=localization.config["rotation_uncertainty_rad"],
        )

    def scene_snapshot(self):
        with self.lock:
            if (
                self.scene is None
                or (
                    self.sync.localization is not None
                    and self.sync.localization.health(self.clock()) is not None
                )
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
            result["localization"] = self.localization_status()
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
            if self.sync.localization_epoch != self.memory_epoch:
                result.update(
                    state="blocked",
                    reason="Localization changed; awaiting new-epoch observations",
                    scene_memory={},
                )
            if result["localization"].get("reason"):
                result.update(state="blocked", reason=result["localization"]["reason"])
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
            if self.memory_epoch != self.sync.localization_epoch:
                self.memory.clear()
                self.scene = None
                self.memory_epoch = self.sync.localization_epoch
                self.status["scene_memory"] = {}
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
            base = bundle.base_poses.get(name)
            poses[name] = self.profile.pose(
                name, bundle.positions[name], base.world_from_base if base else None
            )
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
                translation_errors = [
                    self.profile.pose_uncertainty(
                        n, bundle.positions[n], bundle.joint_gap_s, bundle.skew_s
                    )
                    for n in (left, right)
                ]
                angular_errors = [
                    self.profile.rotation_uncertainty(n, bundle.joint_gap_s) for n in (left, right)
                ]
                world_translation_error, world_rotation_error = 0.0, 0.0
                if bundle.base_poses:
                    for index, name in enumerate((left, right)):
                        base = bundle.base_poses[name]
                        lever = np.linalg.norm(
                            poses[name]["position"] - base.world_from_base[:3, 3]
                        )
                        translation_errors[index] += (
                            base.interpolation_translation_m
                            + lever * base.interpolation_rotation_rad
                        )
                        angular_errors[index] += base.interpolation_rotation_rad
                        world_translation_error = max(
                            world_translation_error,
                            base.translation_uncertainty_m + lever * base.rotation_uncertainty_rad,
                        )
                        world_rotation_error = max(
                            world_rotation_error, base.rotation_uncertainty_rad
                        )
                        # Only identical exposure times justify treating global
                        # localization uncertainty as a common world-frame error.
                        if bundle.skew_s > 1e-9:
                            translation_errors[index] += (
                                base.translation_uncertainty_m
                                + lever * base.rotation_uncertainty_rad
                            )
                            angular_errors[index] += base.rotation_uncertainty_rad
                z, baseline = result.depth, result.baseline_m
                k = result.intrinsics
                ray_factor = np.sqrt(
                    1
                    + (max(abs(k[0, 2]), abs(images[left].shape[1] - k[0, 2])) / k[0, 0]) ** 2
                    + (max(abs(k[1, 2]), abs(images[left].shape[0] - k[1, 2])) / k[1, 1]) ** 2
                )
                # Calibration/pose errors affect triangulation as well as the
                # final world transform, especially at long range/small baseline.
                sigma = (
                    z**2 / (result.focal_px * baseline)
                    + max(translation_errors)
                    + z * sum(translation_errors) / baseline
                    + z**2 * sum(angular_errors) / baseline
                    + z * ray_factor * max(angular_errors)
                    + world_translation_error
                    + z * ray_factor * world_rotation_error
                )
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
        with self.lock:
            if (
                self.sync.fault
                or bundle.localization_epoch != self.sync.localization_epoch
                or self.stop.is_set()
            ):
                self.memory.clear()
                self.scene = None
                return
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
            localization_epoch=bundle.localization_epoch,
            localization_gap_s=max((b.gap_s for b in bundle.base_poses.values()), default=0.0),
            localization={
                n: dict(
                    world_from_base=b.world_from_base.tolist(),
                    gap_s=b.gap_s,
                    translation_uncertainty_m=b.translation_uncertainty_m,
                    rotation_uncertainty_rad=b.rotation_uncertainty_rad,
                    interpolation_translation_m=b.interpolation_translation_m,
                    interpolation_rotation_rad=b.interpolation_rotation_rad,
                    edges=b.edges,
                )
                for n, b in bundle.base_poses.items()
            },
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
            if (
                self.sync.fault
                or self.stop.is_set()
                or bundle.localization_epoch != self.sync.localization_epoch
            ):
                self.memory.clear()
                return
            self.scene = dict(
                timestamp=bundle.timestamp,
                calibration_digest=self.profile.digest,
                world_frame=self.profile.document["world_frame"],
                has_current_depth=bool(clouds),
                localization_epoch=bundle.localization_epoch,
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
