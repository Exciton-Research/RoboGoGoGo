"""Original articulated mobile dual-arm concept, not a vendor CAD replica.

Dimensions, masses, inertias and limits are illustrative design assumptions.
The base has planar support; the mobile session adds tire traction and a stability guard.
Rollover and suspension are not modeled.
"""

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from importlib.resources import files

from .components import ARM_COMPONENTS, mounting_offset, mounting_sign

DESIGN = dict(
    name="RG Mobile Duo · articulated concept",
    architecture="Four steerable wheels, folding torso, two 7-DoF arms, pan/tilt head",
    fidelity="Concept geometry; flat-ground tire dynamics and support guard, no rollover or hardware validation",
    wheel_radius_m=0.14,
    wheelbase_m=0.50,
    track_m=0.68,
    torso_segments_m=[0.44, 0.44],
    arm_dof=7,
)


def mobile_dual_arm_urdf():
    robot = ET.Element("robot", name=DESIGN["name"])

    def link(name, mass=0.0, size=None, center=(0, 0, 0)):
        element = ET.SubElement(robot, "link", name=name)
        if mass:
            inertial = ET.SubElement(element, "inertial")
            ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)))
            ET.SubElement(inertial, "mass", value=str(mass))
            x, y, z = size or (0.1, 0.1, 0.1)
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(mass * (y * y + z * z) / 12),
                iyy=str(mass * (x * x + z * z) / 12),
                izz=str(mass * (x * x + y * y) / 12),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        return element

    def visual(element, shape, dimensions, origin=(0, 0, 0), rpy=(0, 0, 0)):
        item = ET.SubElement(element, "visual")
        ET.SubElement(item, "origin", xyz=" ".join(map(str, origin)), rpy=" ".join(map(str, rpy)))
        geometry = ET.SubElement(item, "geometry")
        if shape == "box":
            ET.SubElement(geometry, shape, size=" ".join(map(str, dimensions)))
        else:
            ET.SubElement(geometry, shape, radius=str(dimensions[0]), length=str(dimensions[1]))

    def joint(
        name,
        parent,
        child,
        kind="fixed",
        origin=(0, 0, 0),
        axis=(0, 0, 1),
        limits=(-2.8, 2.8),
        effort=100.0,
        velocity=1.0,
    ):
        element = ET.SubElement(robot, "joint", name=name, type=kind)
        ET.SubElement(element, "parent", link=parent)
        ET.SubElement(element, "child", link=child)
        ET.SubElement(element, "origin", xyz=" ".join(map(str, origin)))
        if kind != "fixed":
            ET.SubElement(element, "axis", xyz=" ".join(map(str, axis)))
            ET.SubElement(
                element,
                "limit",
                lower=str(limits[0]),
                upper=str(limits[1]),
                effort=str(effort),
                velocity=str(velocity),
            )
            ET.SubElement(element, "dynamics", damping="0.2")

    link("world")
    link("base_x", 0.01)
    joint(
        "base_x_joint", "world", "base_x", "prismatic", axis=(1, 0, 0), limits=(-2, 2), effort=4000
    )
    link("base_y", 0.01)
    joint(
        "base_y_joint", "base_x", "base_y", "prismatic", axis=(0, 1, 0), limits=(-2, 2), effort=4000
    )
    base = link("base_link", 40, (0.76, 0.74, 0.22))
    joint(
        "base_yaw_joint",
        "base_y",
        "base_link",
        "revolute",
        origin=(0, 0, 0.22),
        limits=(-math.pi, math.pi),
        effort=2000,
    )
    for y in (-0.27, 0.27):
        visual(base, "box", (0.76, 0.20, 0.22), origin=(0, y, 0))
    visual(base, "box", (0.22, 0.72, 0.18), origin=(-0.27, 0, 0.02))
    for longitudinal, x in (("front", 0.25), ("rear", -0.25)):
        for side, y in (("left", 0.35), ("right", -0.35)):
            prefix = f"{longitudinal}_{side}"
            link(f"{prefix}_steer", 0.4)
            joint(
                f"{prefix}_steer_joint",
                "base_link",
                f"{prefix}_steer",
                "revolute",
                origin=(x, y, -0.08),
                effort=20,
            )
            wheel = link(f"{prefix}_wheel", 1.0, (0.28, 0.08, 0.28))
            joint(
                f"{prefix}_wheel_joint",
                f"{prefix}_steer",
                f"{prefix}_wheel",
                "continuous",
                axis=(0, 1, 0),
                effort=20,
                velocity=10,
            )
            visual(wheel, "cylinder", (0.14, 0.08), rpy=(math.pi / 2, 0, 0))
    pelvis = link("torso_yaw", 3.0, (0.24, 0.24, 0.12))
    joint(
        "torso_yaw_joint",
        "base_link",
        "torso_yaw",
        "revolute",
        origin=(-0.16, 0, 0.16),
        limits=(-1.0, 1.0),
        effort=500,
    )
    visual(pelvis, "cylinder", (0.12, 0.10))
    lower = link("torso_lower", 5.0, (0.15, 0.18, 0.44), center=(0, 0, 0.22))
    joint(
        "torso_lower_joint",
        "torso_yaw",
        "torso_lower",
        "revolute",
        axis=(0, 1, 0),
        limits=(-0.2, 1.15),
        effort=900,
    )
    visual(lower, "box", (0.15, 0.18, 0.44), origin=(0, 0, 0.22))
    upper = link("torso_upper", 4.0, (0.13, 0.16, 0.44), center=(0, 0, 0.22))
    joint(
        "torso_upper_joint",
        "torso_lower",
        "torso_upper",
        "revolute",
        origin=(0, 0, 0.44),
        axis=(0, 1, 0),
        limits=(-2.3, 0.2),
        effort=900,
    )
    visual(upper, "box", (0.13, 0.16, 0.44), origin=(0, 0, 0.22))
    chest = link("torso", 8.0, (0.28, 0.56, 0.28), center=(0, 0, 0.12))
    joint(
        "torso_pitch_joint",
        "torso_upper",
        "torso",
        "revolute",
        origin=(0, 0, 0.44),
        axis=(0, 1, 0),
        limits=(-1.2, 1.4),
        effort=500,
    )
    visual(chest, "box", (0.28, 0.50, 0.28), origin=(0, 0, 0.12))
    template = ET.fromstring(
        files("robogogogo").joinpath("assets/mobile_arm_chain.urdf").read_text()
    )
    for side, y in (("left", 0.35), ("right", -0.35)):
        for child in template:
            element = deepcopy(child)
            element.set("name", f"{side}_{element.attrib['name']}")
            if element.tag == "joint":
                for tag in ("parent", "child"):
                    item = element.find(tag)
                    item.set("link", f"{side}_{item.attrib['link']}")
                if element.attrib["name"] == f"{side}_waist_joint":
                    element.find("origin").set("xyz", "0 0 0")
                    element.find("axis").set("xyz", "0 1 0")
                if element.attrib["name"] == f"{side}_shoulder_joint":
                    element.find("parent").set("link", f"{side}_upper_roll")
                    element.find("axis").set("xyz", "1 0 0")
                    element.find("origin").set("xyz", "0.06 0 0")
            if element.tag == "link" and element.attrib["name"] == f"{side}_waist":
                element.find("inertial/origin").set("xyz", "0 0 0")
                inertia = element.find("inertial/inertia")
                for diagonal in ("ixx", "iyy", "izz"):
                    inertia.set(diagonal, str(2.0 * 0.07**2 / 6))
                for item in list(element.findall("visual")):
                    element.remove(item)
            if element.tag == "joint":
                short_offsets = {"wrist_pitch": 0.055, "tool_roll": 0.10, "tool": 0.13}
                for suffix, offset in short_offsets.items():
                    if element.attrib["name"] == f"{side}_{suffix}_joint":
                        element.find("origin").set("xyz", f"{offset} 0 0")
            if element.tag == "link":
                segments = {
                    "shoulder": (0.32, 0.055),
                    "elbow": (0.28, 0.065),
                    "wrist_roll": (0.055, 0.043),
                    "wrist_pitch": (0.10, 0.043),
                    "tool_roll": (0.13, 0.04),
                }
                for suffix, (length, radius) in segments.items():
                    if element.attrib["name"] != f"{side}_{suffix}":
                        continue
                    inertial = element.find("inertial")
                    inertial.find("origin").set("xyz", f"{length / 2} 0 0")
                    mass = float(inertial.find("mass").attrib["value"])
                    inertia = inertial.find("inertia")
                    inertia.set("ixx", str(mass * radius**2 / 2))
                    for diagonal in ("iyy", "izz"):
                        inertia.set(diagonal, str(mass * (3 * radius**2 + length**2) / 12))
                    for item in list(element.findall("visual")):
                        element.remove(item)
                    visual(
                        element,
                        "cylinder",
                        (radius, length),
                        origin=(length / 2, 0, 0),
                        rpy=(0, math.pi / 2, 0),
                    )
            robot.append(element)
        joint(f"{side}_mount", "torso", f"{side}_pedestal", origin=(0, y, 0.22))
        link(f"{side}_upper_roll", 0.5)
        joint(
            f"{side}_upper_roll_joint",
            f"{side}_waist",
            f"{side}_upper_roll",
            "revolute",
            origin=(-0.06, 0, 0),
            axis=(0, 0, 1),
            effort=60,
        )
        # Serial Y-Z-X shoulder: torso-mounted lift, cross-axis swing, axial roll.
        # Engineering-9 offsets the swing pivot; earlier trajectories are incompatible;
        # engineering-8 changed joint semantics; older trajectories are incompatible.
    neck = link("head_pan", 0.3)
    joint(
        "head_pan_joint",
        "torso",
        "head_pan",
        "revolute",
        origin=(0, 0, 0.38),
        limits=(-0.7, 0.7),
        effort=10,
    )
    visual(neck, "cylinder", (0.045, 0.08), origin=(0, 0, -0.04))
    head = link("head", 0.6, (0.19, 0.26, 0.18))
    joint(
        "head_pitch_joint",
        "head_pan",
        "head",
        "revolute",
        axis=(0, 1, 0),
        limits=(-0.45, 0.45),
        effort=10,
    )
    visual(head, "box", (0.19, 0.26, 0.18))
    # Add complete arm actuator masses to the structural concept links.
    # Their output-side inertia uses a solid-cylinder approximation; vendor
    # rotor/gear inertia and custom bracket masses are not identified yet.
    for side in ("left", "right"):
        for suffix in ARM_COMPONENTS:
            spec = ARM_COMPONENTS[suffix]
            item = robot.find(f"joint[@name='{side}_{suffix}_joint']")
            item.find("limit").set("effort", str(spec.rated_nm))
            axis = [
                float(v) * mounting_sign(side, suffix) for v in item.find("axis").get("xyz").split()
            ]
            base = [float(v) for v in item.find("origin").get("xyz").split()]
            center = [
                o + a * (mounting_offset(suffix) + spec.length_m / 2) for o, a in zip(base, axis)
            ]
            name = f"{side}_{suffix}_actuator"
            mass = spec.mass_kg
            actuator = link(name, mass)
            inertia = actuator.find("inertial/inertia")
            axial = mass * (spec.diameter_m / 2) ** 2 / 2
            radial = mass * (3 * (spec.diameter_m / 2) ** 2 + spec.length_m**2) / 12
            for tag, a in zip(("ixx", "iyy", "izz"), axis):
                inertia.set(tag, str(axial if abs(a) > 0.5 else radial))
            joint(f"{name}_mount", item.find("parent").get("link"), name, origin=center)
    # Aggregate custom fittings at their actual host frames, rather than
    # placing an approximate bracket mass at each actuator's centre.
    from jax import enable_x64

    from ..urdf import loads
    from .fixture_inertia import add_fixture_inertia
    from .mechanical_covers import motor_cover_specs
    from .packaging import mechanical_assembly_parts

    # Exported fixture dimensions must not depend on the application's default
    # simulation precision. The context restores the caller's configuration.
    with enable_x64(True):
        layout = loads(ET.tostring(robot, encoding="unicode")).model
    fittings = [
        dict(part, body_name=layout.names[part["body"]])
        for part in mechanical_assembly_parts(layout)
    ]
    add_fixture_inertia(robot, fittings + motor_cover_specs(layout))
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode")


def mobile_home(model, posture="standing"):
    import numpy as np

    poses = {"standing": (0.15, -0.30, 0.15), "low": (1.05, -2.10, 1.05), "reach": (0.7, -0.8, 0.1)}
    if posture not in poses:
        raise ValueError("Choose standing, low or reach posture")
    values = dict(zip(("torso_lower", "torso_upper", "torso"), poses[posture]))
    for side in ("left", "right"):
        values.update(
            {
                f"{side}_{name}": value
                for name, value in zip(
                    (
                        "waist",
                        "upper_roll",
                        "shoulder",
                        "elbow",
                        "wrist_roll",
                        "wrist_pitch",
                        "tool_roll",
                    ),
                    (1.2, 0.0, 0.0, -1.35, 0.0, 0.15, 0.0),
                )
            }
        )
    q = np.asarray(model.home).copy()
    for name, value in values.items():
        q[model.indices[model.names.index(name)]] = value
    return q


def mobile_arm_definitions():
    return [
        dict(name=f"{side}_arm", tool=f"{side}_tool", root=f"{side}_pedestal")
        for side in ("left", "right")
    ]
