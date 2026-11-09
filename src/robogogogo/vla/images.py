"""Validation for fixed-camera PNG observations."""

import base64
import struct

from .contracts import CAMERA


def decode_camera_png(data_url: str, camera=CAMERA):
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("Camera observation must be a base64 PNG data URL")
    try:
        data = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except ValueError as error:
        raise ValueError("Invalid camera image encoding") from error
    if len(data) > 512_000 or len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid or oversized PNG observation")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (camera["width"], camera["height"]):
        raise ValueError(
            f"Camera observation must be {camera['width']} × {camera['height']} pixels"
        )
    return data
