"""Versioned wire/storage representation for renderer-derived optical depth."""

import base64
import binascii

import numpy as np


def decode_depth(encoded, camera):
    size = camera["width"] * camera["height"] * 4
    if not isinstance(encoded, str) or len(encoded) > ((size + 2) // 3) * 4:
        raise ValueError("Invalid float32 depth payload size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 depth payload") from error
    if len(raw) != size:
        raise ValueError("Depth dimensions do not match camera")
    depth = np.frombuffer(raw, dtype="<f4").reshape(camera["height"], camera["width"]).copy()
    finite = depth[np.isfinite(depth)]
    if (
        np.isinf(depth).any()
        or (finite < camera.get("near", 0.01) * 0.999).any()
        or (finite > camera.get("far", 50) * 1.001).any()
    ):
        raise ValueError("Depth must be optical Z meters within clipping planes, or NaN")
    return depth
