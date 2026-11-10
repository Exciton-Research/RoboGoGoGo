"""Clearly labeled synthetic SDK telemetry to preview Studio without a real robot.

Run: .venv/bin/python examples/hardware_monitor.py --port 8773
Open /hardware/, select demo_sdk, and connect. This example never contacts hardware.
"""

import argparse
import math
import struct
import time
import zlib

import numpy as np
import uvicorn

from robogogogo.hardware import CameraSample, JointSample, SDKBackend
from robogogogo.studio.api import create_app


def png(image):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    height, width, _ = image.shape
    pixels = b"".join(b"\0" + row.tobytes() for row in image)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class DemoReader:
    def __init__(self):
        self.closed = False
        self.start = time.monotonic()

    def read(self, timeout_s):
        if self.closed:
            return None, ()
        t = time.monotonic() - self.start
        frames = []
        for name, phase in (("demo_head", 0.0), ("demo_wrist", 1.0)):
            yy, xx = np.indices((240, 320))
            image = np.full((240, 320, 3), [28, 45, 62], dtype=np.uint8)
            image[(xx % 40 < 1) | (yy % 40 < 1)] = [65, 85, 103]
            center = 160 + 65 * math.sin(t + phase)
            image[((xx - center) ** 2 + (yy - 120) ** 2) < 32**2] = [68, 183, 175]
            frames.append(CameraSample(name, t, png(image), "image/png", f"{name}_optical"))
        joints = JointSample(
            t,
            ("demo_left_shoulder", "demo_right_shoulder"),
            (0.2 * math.sin(t), 0.2 * math.cos(t)),
            (0.2 * math.cos(t), -0.2 * math.sin(t)),
        )
        return joints, tuple(frames)

    def close(self):
        self.closed = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    app = create_app(hardware_backends={"demo_sdk": lambda: SDKBackend(DemoReader(), period_s=0.1)})
    uvicorn.run(app, host="127.0.0.1", port=args.port)
