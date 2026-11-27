"""Validation for fixed-camera PNG observations."""

import base64
import io
import struct

from PIL import Image

from robogogogo.runtime.contracts import CAMERA

MAX_PNG_BYTES = 512_000


def validate_camera_png(data: bytes, camera=CAMERA):
    """Check image dimensions, chunk integrity, and actual pixel decoding."""
    if (
        not isinstance(data, bytes)
        or not 24 <= len(data) <= MAX_PNG_BYTES
        or data[:8] != b"\x89PNG\r\n\x1a\n"
    ):
        raise ValueError("Invalid or oversized PNG observation")
    width, height = struct.unpack(">II", data[16:24])
    if not 1 <= width <= 640 or not 1 <= height <= 480:
        raise ValueError("PNG observation dimensions exceed the camera limits")
    if (width, height) != (camera["width"], camera["height"]):
        raise ValueError(
            f"Camera observation must be {camera['width']} × {camera['height']} pixels"
        )
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            image.verify()
        # verify() checks chunks/checksums without decoding compressed pixel data.
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            image.load()
    except (OSError, SyntaxError, ValueError, struct.error) as error:
        raise ValueError("Invalid or truncated PNG observation") from error
    return data


def decode_camera_png(data_url: str, camera=CAMERA):
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise ValueError("Camera observation must be a base64 PNG data URL")
    payload = data_url.split(",", 1)[1]
    if len(payload) > ((MAX_PNG_BYTES + 2) // 3) * 4:
        raise ValueError("Invalid or oversized PNG observation")
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError as error:
        raise ValueError("Invalid camera image encoding") from error
    return validate_camera_png(data, camera)
