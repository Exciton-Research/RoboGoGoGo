"""Adapter for a trusted, explicitly read-only manufacturer SDK wrapper."""

import threading
from typing import Protocol

from .contracts import CameraSample, JointSample, TelemetrySink


class TelemetryReader(Protocol):
    def read(self, timeout_s: float) -> tuple[JointSample | None, tuple[CameraSample, ...]]:
        """Read telemetry within timeout_s. Preserve sensor acquisition timestamps."""

    def close(self) -> None:
        """Unblock reads and close the telemetry connection without commanding motion."""


class SDKBackend:
    def __init__(self, reader: TelemetryReader, *, period_s=0.05):
        if not 0.01 <= period_s <= 1:
            raise ValueError("SDK poll period must be 0.01–1 s")
        self.reader, self.period = reader, period_s
        self.stop = threading.Event()
        self.thread = None

    def start(self, sink: TelemetrySink):
        if self.thread is not None:
            raise ValueError("Create a new SDK adapter for each connection")

        def run():
            try:
                while not self.stop.is_set():
                    joint, cameras = self.reader.read(timeout_s=self.period)
                    if joint is not None:
                        sink.joints(joint)
                    for camera in cameras:
                        sink.camera(camera)
                    self.stop.wait(self.period)
            except Exception as error:
                if not self.stop.is_set():
                    sink.error(f"SDK telemetry failed: {error}")

        self.thread = threading.Thread(target=run, daemon=True, name="robot-sdk-telemetry")
        self.thread.start()

    def close(self):
        self.stop.set()
        self.reader.close()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("SDK reader did not stop; implement bounded reads and close()")
