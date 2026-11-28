"""Image and depth checks at the hand and destination; no simulator object state."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from .pretrained import DetectionRejected
from .task import TABLE_Z


def project_bounds(bounds, *, rgb, position, rotation, fovy, **unused):
    low, high = np.asarray(bounds)
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    points = (corners - position) @ np.asarray(rotation).reshape(3, 3)
    if np.any(points[:, 2] >= -0.01):
        raise ValueError("Verification region is behind the inspection camera")
    height, width = rgb.shape[:2]
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    pixels = np.c_[
        width / 2 + focal * points[:, 0] / -points[:, 2],
        height / 2 - focal * points[:, 1] / -points[:, 2],
    ]
    left, top = np.maximum(0, np.floor(pixels.min(0)) - 8).astype(int)
    right, bottom = np.minimum([width, height], np.ceil(pixels.max(0)) + 8).astype(int)
    if right - left < 8 or bottom - top < 8:
        raise ValueError("Verification region is outside the image")
    return int(left), int(top), int(right), int(bottom)


class VisualVerifier:
    def __init__(self, detector, capture):
        self.detector, self.capture = detector, capture
        self.last_image = None
        self.image_revision = 0

    def observe(self, query, bounds):
        from .vision import point_cloud

        confirmed = []
        reasons = []
        for view in ("side", "front", "diagonal", "left"):
            sensor = self.capture(view=view, detail=True)
            rgb = sensor["rgb"]
            left, top, right, bottom = project_bounds(bounds, **sensor)
            candidates = self.detector.detect(rgb[top:bottom, left:right], query, threshold=0.05)
            if not candidates or candidates[0]["score"] < 0.2:
                reasons.append(f"{view}: target not confidently visible")
                continue
            best = candidates[0]
            if len(candidates) > 1 and best["score"] - candidates[1]["score"] < 0.06:
                reasons.append(f"{view}: ambiguous target")
                continue
            box = np.asarray(best["box"]) + [left, top, left, top]
            x0, y0, x1, y1 = np.r_[np.floor(box[:2]), np.ceil(box[2:])].astype(int)
            x0, x1 = np.clip([x0, x1], 0, rgb.shape[1])
            y0, y1 = np.clip([y0, y1], 0, rgb.shape[0])
            cloud = point_cloud(
                sensor["depth"], sensor["position"], sensor["rotation"], sensor["fovy"]
            )[y0:y1, x0:x1].reshape(-1, 3)
            low, high = np.asarray(bounds)
            cloud = cloud[
                np.isfinite(cloud).all(1)
                & np.all(cloud >= low, axis=1)
                & np.all(cloud <= high, axis=1)
            ]
            if len(cloud) < 30:
                reasons.append(f"{view}: insufficient object depth")
                continue
            observed_low, observed_high = np.percentile(cloud, [5, 95], axis=0)
            result = {
                "position": ((observed_low + observed_high) / 2).tolist(),
                "low": observed_low.tolist(),
                "high": observed_high.tolist(),
                "score": float(best["score"]),
                "view": view,
            }
            annotated = Image.fromarray(rgb)
            draw = ImageDraw.Draw(annotated)
            draw.rectangle(box.tolist(), outline="#77bc83", width=4)
            draw.text((box[0], max(0, box[1] - 16)), query, fill="#77bc83")
            output = io.BytesIO()
            annotated.save(output, format="PNG")
            self.last_image = base64.b64encode(output.getvalue()).decode()
            self.image_revision += 1
            for other in confirmed:
                if np.linalg.norm(np.asarray(result["position"]) - other["position"]) < 0.04:
                    return {
                        **result,
                        "position": np.mean(
                            [result["position"], other["position"]], axis=0
                        ).tolist(),
                        "views": [other["view"], view],
                    }
            confirmed.append(result)
        raise DetectionRejected(
            "Could not verify the named object from two camera views. " + "; ".join(reasons[:2])
        )

    def lift(self, query, *, expected, opening):
        if not 0.008 < opening < 0.078:
            raise DetectionRejected("Finger opening indicates an empty or unstable grasp")
        expected = np.asarray(expected)
        low = expected - [0.12, 0.12, 0.14]
        high = expected + [0.12, 0.12, 0.14]
        low[2] = max(TABLE_Z + 0.07, low[2])
        result = self.observe(query, [low, high])
        if np.linalg.norm(np.asarray(result["position"]) - expected) > 0.09:
            raise DetectionRejected("Named object is not at the measured grasp location")
        return result

    def placement(self, query, *, destination, tool, opening):
        if opening < 0.065 or np.linalg.norm(np.asarray(tool) - destination) < 0.12:
            raise DetectionRejected("Gripper has not released and withdrawn from the destination")
        x, y = destination[:2]
        result = self.observe(
            query, [[x - 0.11, y - 0.11, TABLE_Z + 0.008], [x + 0.11, y + 0.11, TABLE_Z + 0.23]]
        )
        if np.max(np.abs(np.asarray(result["position"])[:2] - [x, y])) > 0.055:
            raise DetectionRejected("Named object is outside the destination cell")
        if result["low"][2] > TABLE_Z + 0.045:
            raise DetectionRejected("Object has not visibly reached the table surface")
        return result
