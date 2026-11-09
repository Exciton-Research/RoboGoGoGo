"""Strict URDF subset loader written with Python's XML parser.

Fixed-base trees; fixed, revolute, continuous and prismatic joints. Meshes,
floating joints, mimic joints, transmissions, and nonspherical collisions are
rejected rather than silently simulated incorrectly. Visual primitives are
returned as metadata; they do not create collisions automatically.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .spatial import Model, ModelBuilder, rpy_matrix


@dataclass(frozen=True)
class Robot:
    name: str
    model: Model
    joint_names: tuple[str, ...]
    visuals: tuple[dict, ...]


def _numbers(text, count, default):
    result = [float(x) for x in text.split()] if text else list(default)
    if len(result) != count or not np.isfinite(result).all():
        raise ValueError(f"expected {count} finite numbers, got {text!r}")
    return result


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (_numbers(origin.get("xyz"), 3, (0, 0, 0)), _numbers(origin.get("rpy"), 3, (0, 0, 0)))


def loads(text, *, dt=0.002, gravity=(0.0, 0.0, -9.81)):
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    if root.findall("transmission") or root.findall("gazebo"):
        raise ValueError("transmission/gazebo extensions are not supported")
    links = {link.get("name"): link for link in root.findall("link")}
    if not links or None in links or len(links) != len(root.findall("link")):
        raise ValueError("URDF needs uniquely named links")
    children, incoming, joint_names = {}, {}, set()
    for joint in root.findall("joint"):
        name, kind = joint.get("name"), joint.get("type")
        if not name or name in joint_names:
            raise ValueError("joints need unique names")
        joint_names.add(name)
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"unsupported joint type: {kind}")
        if joint.find("mimic") is not None:
            raise ValueError("mimic joints are not supported")
        try:
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
        except (AttributeError, KeyError) as error:
            raise ValueError("joint requires parent and child links") from error
        if parent not in links or child not in links or child in incoming:
            raise ValueError("invalid parent/child or multiple parents")
        incoming[child] = joint
        children.setdefault(parent, []).append(child)
    roots = set(links) - set(incoming)
    if len(roots) != 1:
        raise ValueError("URDF must contain one rooted tree")
    builder = ModelBuilder(dt=dt, gravity=gravity)
    mapping, visuals, moving_names = {}, [], []

    def visit(name, parent=-1):
        if name in mapping:
            raise ValueError("cycle in URDF")
        link = links[name]
        inertial = link.find("inertial")
        if inertial is None:
            if parent != -1:
                raise ValueError(
                    f"link {name}: explicit inertial required (including fixed descendants)"
                )
            mass, com, inertia = 0.0, [0.0, 0.0, 0.0], np.zeros((3, 3))
        else:
            try:
                mass = float(inertial.find("mass").attrib["value"])
                element = inertial.find("inertia")
                xx, yy, zz, xy, xz, yz = [
                    float(element.attrib[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ]
            except (AttributeError, KeyError) as error:
                raise ValueError(f"link {name}: incomplete inertia") from error
            com, rpy = _origin(inertial)
            rotation = np.asarray(rpy_matrix(rpy))
            inertia = rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
        kwargs = dict(parent=parent, mass=mass, com=com, inertia=inertia)
        if name not in incoming:
            kwargs["joint"] = "fixed"
        else:
            joint = incoming[name]
            kind = joint.attrib["type"]
            xyz, rpy = _origin(joint)
            axis = joint.find("axis")
            kwargs.update(
                joint=kind,
                origin=xyz,
                rpy=rpy,
                axis=_numbers(None if axis is None else axis.get("xyz"), 3, (1, 0, 0)),
            )
            dynamics = joint.find("dynamics")
            kwargs["damping"] = float(dynamics.get("damping", "0")) if dynamics is not None else 0.0
            if dynamics is not None and float(dynamics.get("friction", "0")) != 0:
                raise ValueError("URDF joint dry friction is not supported")
            if kind != "fixed":
                limit = joint.find("limit")
                if limit is None or "effort" not in limit.attrib:
                    raise ValueError(f"joint {joint.attrib['name']}: effort limit required")
                kwargs["effort"] = float(limit.attrib["effort"])
                if kind != "continuous":
                    if "lower" not in limit.attrib or "upper" not in limit.attrib:
                        raise ValueError("bounded joints require lower and upper limits")
                    low, high = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                    kwargs.update(limits=(low, high), home=float(np.clip(0.0, low, high)))
                # Velocity is metadata in URDF; this engine does not hard-clip qd.
                moving_names.append(joint.attrib["name"])
        body = builder.add_link(name, **kwargs)
        mapping[name] = body
        for collision in link.findall("collision"):
            xyz, _ = _origin(collision)
            geometry = collision.find("geometry")
            sphere = None if geometry is None else geometry.find("sphere")
            if sphere is None or len(geometry) != 1:
                raise ValueError(f"link {name}: only sphere collision geometry is supported")
            builder.add_sphere(body, xyz, float(sphere.attrib["radius"]))
        for visual in link.findall("visual"):
            xyz, rpy = _origin(visual)
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("visual requires one geometry")
            shape = geometry[0]
            if shape.tag not in ("box", "sphere", "cylinder"):
                raise ValueError(f"unsupported visual: {shape.tag}")
            dimensions = (
                _numbers(shape.get("size"), 3, ())
                if shape.tag == "box"
                else [float(shape.attrib["radius"])]
                if shape.tag == "sphere"
                else [float(shape.attrib["radius"]), float(shape.attrib["length"])]
            )
            if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
                raise ValueError("visual dimensions must be positive and finite")
            visuals.append(
                dict(body=body, shape=shape.tag, dimensions=dimensions, origin=xyz, rpy=rpy)
            )
        for child in children.get(name, []):
            visit(child, body)

    visit(next(iter(roots)))
    if len(mapping) != len(links):
        raise ValueError("disconnected links or cycle in URDF")
    return Robot(root.get("name", "robot"), builder.build(), tuple(moving_names), tuple(visuals))


def load(path, **kwargs):
    return loads(Path(path).read_text(), **kwargs)
