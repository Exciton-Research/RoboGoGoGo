"""Catalog image matching and calibrated RGB-D localization, without scene labels.

Reference images come from isolated library scans, never from the live scene.
This is a closed-catalog baseline, not an open-vocabulary detector.
"""

import base64
import io
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .objects import CATALOG, directory, require_object
from .task import TABLE_Z, add_scan

VISION_WIDTH, VISION_HEIGHT = 640, 384
CAMERA_POSITION = np.array([0.50, -0.85, 1.40])
CAMERA_LOOK_AT = np.array([0.50, -0.10, 0.78])


def inspection_camera(view="front"):
    """Fixed calibrated views; neither depends on the target or simulator poses."""
    if view not in ("front", "side"):
        raise ValueError("Inspection view must be front or side")
    position, look_at = (
        (np.array([1.08, -0.17, 1.25]), np.array([0.46, -0.16, 0.80]))
        if view == "side"
        else (CAMERA_POSITION, CAMERA_LOOK_AT)
    )
    offset = position - look_at
    camera = mujoco.MjvCamera()
    camera.lookat[:] = look_at
    camera.distance = np.linalg.norm(offset)
    camera.azimuth = np.rad2deg(np.arctan2(-offset[1], -offset[0]))
    camera.elevation = -np.rad2deg(np.arctan2(offset[2], np.linalg.norm(offset[:2])))
    return camera


def point_cloud(depth, position, rotation, fovy):
    """Back-project optical-axis depth using an OpenGL camera-to-world rotation."""
    height, width = depth.shape
    v, u = np.indices(depth.shape)
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    camera = np.stack(
        ((u + 0.5 - width / 2) * depth / focal, -(v + 0.5 - height / 2) * depth / focal, -depth),
        axis=-1,
    )
    return camera @ np.asarray(rotation).reshape(3, 3).T + position


def descriptor(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [24, 8], [0, 180, 0, 256])
    hist = np.sqrt(hist.ravel() / max(1, hist.sum()))
    return hist


def proposals(rgb, depth, position, rotation, fovy, *, with_appearance=True):
    points = point_cloud(depth, position, rotation, fovy)
    x, y, z = points.transpose(2, 0, 1)
    # A calibrated pickup workspace, not slots or object-specific locations.
    mask = (
        np.isfinite(depth)
        & (depth > 0)
        & (x > 0.19)
        & (x < 0.84)
        & (y > -0.39)
        & (y < 0.035)
        & (z > TABLE_Z + 0.008)
        & (z < TABLE_Z + 0.23)
    )
    # Separate touching silhouettes where their observed surface heights jump.
    jumps = np.zeros(mask.shape, dtype=bool)
    for axis in (0, 1):
        difference = np.abs(np.diff(z, axis=axis)) > 0.012
        if axis == 0:
            jumps[1:] |= difference
            jumps[:-1] |= difference
        else:
            jumps[:, 1:] |= difference
            jumps[:, :-1] |= difference
    mask &= ~jumps
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    result = []
    for label in range(1, count):
        left, top, width, height, area = stats[label]
        if area < 35 or width < 4 or height < 4:
            continue
        selected = labels == label
        cloud = points[selected]
        low, high = np.percentile(cloud, [1, 99], axis=0)
        result.append(
            {
                "box": [int(left), int(top), int(left + width), int(top + height)],
                "xy": (low[:2] + high[:2]) / 2,
                "size": np.r_[np.sort(high[:2] - low[:2]), high[2] - TABLE_Z],
                "feature": descriptor(rgb, selected) if with_appearance else None,
                "cloud": cloud,
            }
        )
    return result


def workspace_crop(*, rgb, depth, position, rotation, fovy):
    """Pixel bounds of visible pickup geometry, without object IDs or templates."""
    regions = proposals(rgb, depth, position, rotation, fovy, with_appearance=False)
    if not regions:
        raise ValueError("No visible objects in the calibrated pickup workspace")
    boxes = np.asarray([region["box"] for region in regions])
    low = np.maximum(0, boxes[:, :2].min(axis=0) - 16)
    high = np.minimum([rgb.shape[1], rgb.shape[0]], boxes[:, 2:].max(axis=0) + 16)
    return [int(value) for value in np.r_[low, high]]


def localize_box(box, *, rgb, depth, position, rotation, fovy):
    """Refine a text detector's pixel box using depth geometry, without object priors."""
    from .pretrained import box_iou

    regions = proposals(rgb, depth, position, rotation, fovy, with_appearance=False)
    ranked = sorted(
        ((box_iou(box, region["box"]), region) for region in regions),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.3:
        raise ValueError("Detection has no reliable depth region in the pickup workspace")
    if len(ranked) > 1 and ranked[1][0] > ranked[0][0] * 0.75:
        raise ValueError("Detection overlaps multiple depth regions; no motion started")
    region = ranked[0][1]
    left, top, right, bottom = region["box"]
    focal = depth.shape[0] / (2 * np.tan(np.deg2rad(fovy) / 2))
    ray = np.asarray(rotation).reshape(3, 3) @ np.array(
        [
            ((left + right) / 2 - depth.shape[1] / 2) / focal,
            -((top + bottom) / 2 - depth.shape[0] / 2) / focal,
            -1,
        ]
    )
    # Upright resting-object assumption; observed height replaces catalog dimensions.
    height = region["size"][2]
    if not 0.015 <= height <= 0.22 or abs(ray[2]) < 1e-6:
        raise ValueError("Depth cannot support an upright tabletop pickup")
    center_z = TABLE_Z + height / 2
    pick = np.asarray(position) + ray * ((center_z - position[2]) / ray[2])
    grasp = pick.copy()
    # Objects taller than the finger clearance need an upper grasp, otherwise
    # the hand housing strikes their top. Grip 25 mm below the visible top so
    # the pads retain contact during lowering. Retain the center for placement.
    if height > 0.08:
        grasp[2] = TABLE_Z + height - 0.025
        upper = region["cloud"][region["cloud"][:, 2] > TABLE_Z + height - 0.01]
        if len(upper) < 12:
            raise ValueError("The upper grasp surface has insufficient visible depth")
        low, high = np.percentile(upper, [3, 97], axis=0)
        grasp[:2] = (low[:2] + high[:2]) / 2
    return pick, region["box"], grasp


class CatalogVision:
    def __init__(self, cache):
        self.cache = cache
        self.references = {}

    def _reference(self, identifier):
        if identifier not in CATALOG or not CATALOG[identifier].get("reference_matching", True):
            raise ValueError(
                "This object is excluded from catalog matching; use pretrained detection"
            )
        if identifier in self.references:
            return self.references[identifier]
        spec = require_object(self.cache, identifier)
        root = ET.fromstring(
            '<mujoco><visual><global offwidth="640" offheight="384"/></visual>'
            "<asset/><worldbody>"
            '<light pos="1.5 -1.5 4" dir="-1 1 -3" diffuse="0.7 0.7 0.7" '
            'ambient="0.15 0.15 0.15"/></worldbody></mujoco>'
        )
        body = ET.SubElement(root.find("worldbody"), "body", name="reference")
        add_scan(root, body, spec, directory(self.cache, identifier))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        features = []
        with mujoco.Renderer(model, height=VISION_HEIGHT, width=VISION_WIDTH) as renderer:
            for x, y in ((0.28, -0.26), (0.48, -0.20), (0.72, -0.10)):
                model.body_pos[body_id := model.body("reference").id] = [
                    x,
                    y,
                    TABLE_Z + spec["dimensions"][2] / 2,
                ]
                for yaw in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    model.body_quat[body_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=inspection_camera(), scene_option=options)
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    features.append(descriptor(rgb, depth < 2))
        result = (spec, np.asarray(features))
        self.references[identifier] = result
        return result

    def find(self, identifier, *, rgb, depth, position, rotation, fovy, catalog_ids):
        """Accept pixels/calibration/catalog only; no env, instance IDs or object poses."""
        allowed = [
            key
            for key in catalog_ids
            if key in CATALOG and CATALOG[key].get("reference_matching", True)
        ]
        if identifier not in allowed:
            raise ValueError("This object has no catalog references; use pretrained detection")
        references = {key: self._reference(key) for key in allowed}
        candidates = proposals(rgb, depth, position, rotation, fovy)
        ranked = []
        for candidate in candidates:
            scores = {}
            for key, (spec, features) in references.items():
                similarity = float(np.max(features @ candidate["feature"]))
                expected = np.r_[np.sort(spec["dimensions"][:2]), spec["dimensions"][2]]
                shape_error = np.mean(
                    np.abs(np.log(np.maximum(candidate["size"], 0.005) / expected))
                )
                scores[key] = similarity - 0.18 * shape_error
            target_score = scores[identifier]
            other_score = max((v for k, v in scores.items() if k != identifier), default=0)
            ranked.append((target_score, target_score - other_score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.65 or ranked[0][1] < 0.025:
            raise ValueError(
                "Camera could not confidently identify the named object. Clear the view and retry."
            )
        score, margin, match = ranked[0]
        if len(ranked) > 1 and score - ranked[1][0] < 0.025:
            raise ValueError("Camera found ambiguous matches; collection was not started")
        spec = references[identifier][0]
        pick = np.r_[match["xy"], TABLE_Z + spec["dimensions"][2] / 2]
        if spec["category"] == "Fruit":
            # A visible hemisphere's centroid is biased toward the camera.
            # Fit the catalog ellipsoid with resting height constrained by the table.
            cloud = match["cloud"] - pick
            radii = np.asarray(spec["dimensions"]) / 2
            design = np.column_stack((2 * cloud[:, :2] / radii[:2] ** 2, np.ones(len(cloud))))
            fit = np.linalg.lstsq(design, np.sum((cloud / radii) ** 2, axis=1), rcond=None)[0]
            if np.linalg.norm(fit[:2]) > 0.025:
                raise ValueError("Camera surface fit is unreliable; collection was not started")
            pick[:2] += fit[:2]
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(match["box"], outline="#ffb340", width=3)
        draw.text((match["box"][0], max(0, match["box"][1] - 14)), spec["name"], fill="#ffb340")
        output = io.BytesIO()
        annotated.save(output, format="PNG")
        return {
            "name": spec["name"],
            "position": pick.tolist(),
            "box": match["box"],
            "score": score,
            "margin": margin,
            "image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
            "method": "Catalog RGB matching + calibrated depth; known upright object dimensions",
        }
