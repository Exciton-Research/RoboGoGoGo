"""Thread-safe, bounded telemetry sessions shared by Studio hardware adapters."""

import math
import re
import threading
import time
from dataclasses import asdict
from typing import Callable

from .contracts import CameraSample, JointSample, ReadOnlyBackend, TelemetrySink


class HardwareService:
    def __init__(
        self,
        backends: dict[str, Callable[[], ReadOnlyBackend]],
        *,
        stale_after=2.0,
        clock=time.monotonic,
        perception_profiles=None,
        audit_dir=None,
    ):
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("Telemetry freshness timeout must be positive")
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) for name in backends):
            raise ValueError("Invalid backend name")
        self.backends = dict(backends)
        self.profiles = dict(perception_profiles or {})
        if not set(self.profiles).issubset(self.backends):
            raise ValueError("Calibration profiles must name registered hardware backends")
        self.audit_dir = audit_dir
        self.perception = None
        self.stale_after, self.clock = stale_after, clock
        self.lock, self.lifecycle = threading.RLock(), threading.Lock()
        self.backend = None
        self.name = None
        self.generation = 0
        self.revision = 0
        self.joint = None
        self.cameras = {}
        self.error = None

    def connect(self, name):
        with self.lifecycle:
            if name not in self.backends:
                raise ValueError("Unknown registered hardware backend")
            self._close()
            with self.lock:
                self.name = name
                generation = self.generation
            sink = TelemetrySink(
                lambda value: self._receive(generation, value),
                lambda value: self._receive(generation, value),
                lambda message: self._error(generation, message),
                lambda info: self._calibration(generation, info),
                lambda pose: self._localization(generation, pose),
            )
            try:
                backend = self.backends[name]()
                with self.lock:
                    self.backend = backend
                if name in self.profiles:
                    from .perception import HardwarePerception

                    pipeline = HardwarePerception(
                        self.profiles[name], audit_dir=self.audit_dir, clock=self.clock
                    )
                    with self.lock:
                        self.perception = pipeline
                    pipeline.start()
                backend.start(sink)
            except Exception as error:
                self._close()
                with self.lock:
                    self.error = str(error)
                raise ValueError(f"Hardware connection failed: {error}") from error
            return self.snapshot()

    def _localization(self, generation, pose):
        with self.lock:
            if generation == self.generation and self.perception:
                self.perception.ingest(pose)

    def _calibration(self, generation, info):
        with self.lock:
            if generation == self.generation and self.perception:
                self.perception.calibration_info(info)

    def _error(self, generation, message):
        with self.lock:
            if generation == self.generation:
                self.error = str(message)[:500]
                if self.perception:
                    self.perception.invalidate(self.error)

    def _receive(self, generation, sample):
        try:
            if not math.isfinite(sample.timestamp) or sample.timestamp < 0:
                raise ValueError("Invalid acquisition timestamp")
            if isinstance(sample, JointSample):
                n = len(sample.names)
                if (
                    not 1 <= n <= 256
                    or len(set(sample.names)) != n
                    or any(
                        not isinstance(name, str) or not 1 <= len(name) <= 128
                        for name in sample.names
                    )
                    or len(sample.position) != n
                    or len(sample.velocity) not in (0, n)
                    or len(sample.effort) not in (0, n)
                    or not all(
                        math.isfinite(v)
                        for v in (*sample.position, *sample.velocity, *sample.effort)
                    )
                ):
                    raise ValueError("Invalid named joint telemetry")
            elif isinstance(sample, CameraSample):
                if (
                    not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", sample.name)
                    or not isinstance(sample.data, bytes)
                    or not 0 < len(sample.data) <= 8 * 1024 * 1024
                    or sample.media_type not in ("image/png", "image/jpeg")
                    or (
                        sample.media_type == "image/png"
                        and not sample.data.startswith(b"\x89PNG\r\n\x1a\n")
                    )
                    or (
                        sample.media_type == "image/jpeg"
                        and not sample.data.startswith(b"\xff\xd8\xff")
                    )
                ):
                    raise ValueError("Camera frames must be bounded PNG or JPEG images")
            else:
                raise ValueError("Unsupported telemetry sample")
            with self.lock:
                if generation != self.generation:
                    return
                old = (
                    self.joint if isinstance(sample, JointSample) else self.cameras.get(sample.name)
                )
                if self.perception:
                    self.perception.ingest(sample)
                # Replayed samples cannot keep a dead sensor looking live.
                if old and sample.timestamp <= old[0].timestamp:
                    return
                self.revision += 1
                item = (sample, self.clock(), self.revision)
                if isinstance(sample, JointSample):
                    self.joint = item
                else:
                    if sample.name not in self.cameras and len(self.cameras) >= 8:
                        raise ValueError("At most eight camera streams are supported")
                    self.cameras[sample.name] = item
        except (ValueError, TypeError, AttributeError) as error:
            self._error(generation, str(error))

    def snapshot(self):
        with self.lock:
            now = self.clock()
            connected = self.backend is not None

            def metadata(item):
                sample, received, revision = item
                age = max(0, now - received)
                return dict(
                    timestamp=sample.timestamp,
                    frame_id=sample.frame_id,
                    age_s=age,
                    stale=not connected or bool(self.error) or age > self.stale_after,
                    revision=revision,
                )

            joint = asdict(self.joint[0]) if self.joint else None
            if joint is not None:
                joint.update(metadata(self.joint))
            cameras = [
                dict(name=name, media_type=item[0].media_type, **metadata(item))
                for name, item in sorted(self.cameras.items())
            ]
            streams = ([joint] if joint else []) + cameras
            status = (
                "fault"
                if self.error
                else "disconnected"
                if not connected
                else "waiting"
                if not streams
                else "stale"
                if all(s["stale"] for s in streams)
                else "streaming"
            )
            return dict(
                backends=sorted(self.backends),
                backend=self.name,
                read_only=True,
                status=status,
                error=self.error,
                joints=joint,
                cameras=cameras,
                session=self.generation,
                perception=self.perception.snapshot()
                if self.perception
                else dict(state="not_configured", reason="No measured calibration profile active"),
            )

    def scene(self):
        with self.lock:
            if not self.perception or self.error:
                raise ValueError("No active calibrated hardware perception")
            return self.perception.scene_snapshot()

    def image(self, name):
        with self.lock:
            if self.backend is None or name not in self.cameras:
                raise FileNotFoundError("Camera frame unavailable")
            return self.cameras[name][0]

    def _close(self):
        with self.lock:
            backend, self.backend = self.backend, None
            pipeline, self.perception = self.perception, None
            self.generation += 1
            self.name, self.joint, self.error = None, None, None
            self.cameras = {}
        errors = []
        for resource in (backend, pipeline):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            with self.lock:
                self.error = "Hardware cleanup failed: " + "; ".join(errors)
            raise ValueError(self.error)

    def close(self):
        with self.lifecycle:
            self._close()
