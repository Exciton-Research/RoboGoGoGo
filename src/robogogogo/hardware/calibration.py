"""Versioned measured calibration and hardware-only URDF forward kinematics.

Visuals, inertias and collisions are irrelevant here and never enter dynamics.
Floating and mimic joints are rejected rather than assigning invented poses.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def transform(value):
    value = np.asarray(value, float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1)
    ):
        raise ValueError("Extrinsics must be finite rigid camera-to-parent transforms")
    return value


def axis_rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


class KinematicTree:
    def __init__(self, xml):
        root = ET.fromstring(xml)
        if root.tag != "robot":
            raise ValueError("Expected a URDF robot")
        self.links = {link.get("name") for link in root.findall("link")}
        if None in self.links or len(self.links) != len(root.findall("link")):
            raise ValueError("URDF links must be uniquely named")
        self.joints = {}
        incoming = set()
        for joint in root.findall("joint"):
            name, kind = joint.get("name"), joint.get("type")
            if (
                kind not in ("fixed", "revolute", "continuous", "prismatic")
                or joint.find("mimic") is not None
            ):
                raise ValueError(
                    "Hardware kinematics supports fixed/revolute/continuous/prismatic joints without mimic"
                )
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                raise ValueError("Joint requires parent and child links")
            parent, child = parent.get("link"), child.get("link")
            if (
                not name
                or name in self.joints
                or child in incoming
                or parent not in self.links
                or child not in self.links
            ):
                raise ValueError("Invalid URDF joint tree")
            incoming.add(child)
            origin = joint.find("origin")
            xyz = np.fromstring(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            rpy = np.fromstring(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
            if (
                any(v.shape != (3,) or not np.isfinite(v).all() for v in (xyz, rpy, axis))
                or np.linalg.norm(axis) < 1e-9
            ):
                raise ValueError("Invalid URDF joint origin/axis")
            offset = np.eye(4)
            offset[:3, 3] = xyz
            offset[:3, :3] = (
                axis_rotation([0, 0, 1], rpy[2])
                @ axis_rotation([0, 1, 0], rpy[1])
                @ axis_rotation([1, 0, 0], rpy[0])
            )
            self.joints[name] = dict(
                parent=parent,
                child=child,
                kind=kind,
                origin=offset,
                axis=axis / np.linalg.norm(axis),
            )
        roots = self.links - incoming
        if len(roots) != 1:
            raise ValueError("Hardware URDF must have one fixed base")
        self.root = roots.pop()
        self.names = tuple(n for n, j in self.joints.items() if j["kind"] != "fixed")
        self.poses(dict.fromkeys(self.names, 0.0))

    def poses(self, positions):
        if (
            not set(self.names).issubset(positions)
            or not np.isfinite([positions[n] for n in self.names]).all()
        ):
            raise ValueError("Missing or invalid measured joint positions")
        result, pending = {self.root: np.eye(4)}, dict(self.joints)
        while pending:
            progress = False
            for name, joint in list(pending.items()):
                if joint["parent"] not in result:
                    continue
                motion = np.eye(4)
                if joint["kind"] in ("revolute", "continuous"):
                    motion[:3, :3] = axis_rotation(joint["axis"], positions[name])
                elif joint["kind"] == "prismatic":
                    motion[:3, 3] = joint["axis"] * positions[name]
                result[joint["child"]] = result[joint["parent"]] @ joint["origin"] @ motion
                del pending[name]
                progress = True
            if not progress:
                raise ValueError("Disconnected or cyclic hardware URDF")
        if set(result) != self.links:
            raise ValueError("Disconnected hardware links")
        return result


class CalibrationProfile:
    def __init__(self, document, robot_xml):
        # JSON round-trip prevents subsequent caller mutation and rejects NaN.
        self.document = json.loads(json.dumps(document, allow_nan=False))
        d = self.document
        if d.get("schema_version") != 1 or not d.get("revision") or d.get("fixed_base") is not True:
            raise ValueError("A versioned measured profile with fixed_base=true is required")
        if not d.get("world_frame") or not d.get("clock_domain") or not d.get("calibrated_at"):
            raise ValueError("Profile requires world frame, clock domain and calibration date")
        self.tree = KinematicTree(robot_xml)
        self.world_from_base = transform(d["world_from_base"])
        self.speed_limits = d["joint_speed_limits"]
        if set(self.speed_limits) != set(self.tree.names) or any(
            not np.isfinite(v) or not 0 < v <= 100 for v in self.speed_limits.values()
        ):
            raise ValueError(
                "Measured robot joint speed limits are required for every movable joint"
            )
        self.max_skew = float(d.get("max_skew_s", 0.01))
        self.max_gap = float(d.get("max_joint_gap_s", 0.05))
        if not 0 < self.max_skew <= 0.03 or not 0 < self.max_gap <= 0.1:
            raise ValueError("Invalid synchronization tolerances")
        self.cameras = d["cameras"]
        if not isinstance(self.cameras, dict) or not 2 <= len(self.cameras) <= 8:
            raise ValueError("Configure two to eight calibrated cameras")
        for name, camera in self.cameras.items():
            k = np.asarray(camera["intrinsics"], float)
            distortion = np.asarray(camera["distortion"], float)
            if (
                k.shape != (3, 3)
                or not np.isfinite(k).all()
                or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or abs(k[0, 1]) + abs(k[1, 0]) > 1e-9
                or distortion.shape not in ((4,), (5,), (8,))
                or not np.isfinite(distortion).all()
                or camera.get("distortion_model") != "plumb_bob"
                or not 32 <= camera["width"] <= 1920
                or not 32 <= camera["height"] <= 1080
                or not camera.get("frame_id")
                or camera["mount_link"] not in self.tree.links
            ):
                raise ValueError(f"Invalid camera calibration: {name}")
            transform(camera["mount_from_optical"])
            if (
                not 0 <= camera["reprojection_error_px"] <= 1
                or not 0 < camera["extrinsic_uncertainty_m"] <= 0.01
            ):
                raise ValueError(
                    "Calibration quality must be supplied and within acceptance bounds"
                )
            if (
                camera["clock_domain"] != d["clock_domain"]
                or not np.isfinite(camera["time_offset_s"])
                or not 0 <= camera["time_uncertainty_s"] <= self.max_skew / 4
            ):
                raise ValueError(
                    "Cameras must share the declared clock domain with bounded measured time offsets"
                )
        self.pairs = tuple(tuple(pair) for pair in d["stereo_pairs"])
        if not 1 <= len(self.pairs) <= 4 or any(
            len(p) != 2 or p[0] == p[1] or any(n not in self.cameras for n in p) for p in self.pairs
        ):
            raise ValueError("Invalid calibrated stereo pairs")
        if set(n for pair in self.pairs for n in pair) != set(self.cameras):
            raise ValueError("Every configured camera must belong to a stereo pair")
        self.digest = hashlib.sha256(
            (json.dumps(d, sort_keys=True) + robot_xml).encode()
        ).hexdigest()

    @classmethod
    def load(cls, filename):
        path = Path(filename)
        document = json.loads(path.read_text())
        return cls(document, (path.parent / document["robot_urdf"]).read_text())

    def pose(self, name, positions):
        camera = self.cameras[name]
        pose = (
            self.world_from_base
            @ self.tree.poses(positions)[camera["mount_link"]]
            @ transform(camera["mount_from_optical"])
        )
        return dict(position=pose[:3, 3], rotation=pose[:3, :3])
