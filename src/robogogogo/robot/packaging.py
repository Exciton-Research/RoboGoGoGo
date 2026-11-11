"""Shared catalog arm dimensions and custom fitting collision geometry.

Vendor mounting holes and connector clocking need CAD confirmation. Moving grouped spans, their link fixtures and panel connectors share this model.
Complete electrical branches and vendor drive leads remain unqualified.
"""

import numpy as np

from ..planning.collision import Capsule


def shoulder_packages(model):
    from .components import ARM_COMPONENTS, mounting_offset, mounting_sign

    result = []
    for side in ("left", "right"):
        for suffix in ARM_COMPONENTS:
            body = model.names.index(f"{side}_{suffix}")
            part = ARM_COMPONENTS[suffix]
            axis = (
                np.asarray(model.orientation[body])
                @ np.asarray(model.axis[body])
                * mounting_sign(side, suffix)
            )
            origin = np.asarray(model.origin[body])
            offset = mounting_offset(suffix)
            result.append(
                dict(
                    name=f"{side}_{suffix}_drive_package",
                    body=model.parents[body],
                    start=(origin + axis * offset).tolist(),
                    end=(origin + axis * (offset + part.length_m)).tolist(),
                    radius=part.diameter_m / 2,
                    mounting_offset=offset,
                    cover_allowance_m=0.004,
                    shoulder_axis={"waist": "yaw", "upper_roll": "pitch", "shoulder": "roll"}.get(
                        suffix
                    ),
                    component=part.to_dict(),
                    provenance="manufacturer external dimensions; custom mounting interfaces",
                )
            )
    return result


def packaging_capsules(model):
    # A sphere enclosing each complete finite cylinder. More conservative than
    # vendor CAD, but it cannot falsely clear intersecting module bodies.
    return [
        Capsule(
            p["name"],
            p["body"],
            tuple((np.array(p["start"]) + p["end"]) / 2),
            tuple((np.array(p["start"]) + p["end"]) / 2),
            float(
                np.hypot(
                    p["radius"] + p["cover_allowance_m"],
                    np.linalg.norm(np.array(p["end"]) - p["start"]) / 2 + p["cover_allowance_m"],
                )
            ),
        )
        for p in shoulder_packages(model)
    ]


def arm_cable_passages(model):
    # Straight center bores in rigid links; no invented flexible cable crossing a joint.
    return [
        dict(
            name=f"{side}_{link}_cable_passage",
            body=model.names.index(f"{side}_{link}"),
            start=[0.025, 0, 0],
            end=[length - 0.025, 0, 0],
            radius=0.018,
            bundle_radius=0.015,
            status="rigid channel; grouped wrist span modeled; intermediate electrical branches unresolved",
        )
        for side in ("left", "right")
        for link, length in (("shoulder", 0.32), ("elbow", 0.28))
    ]


def assembly_parts(model, *, electrical=True):
    """Custom mounting saddles and connector installation envelopes in parent frames.

    Rails and end ties form a cage around each module. Dimensions are our design,
    not vendor mounting-hole CAD; rear connector placement needs vendor CAD confirmation.
    """
    result = []
    for drive in shoulder_packages(model):
        a, b = np.array(drive["start"]), np.array(drive["end"])
        axis = (b - a) / np.linalg.norm(b - a)
        radial = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.5 else np.array([0.0, 1, 0])
        other = np.cross(axis, radial)
        # The lift yoke opens toward the descending upper arm, not across it.
        if drive["shoulder_axis"] == "pitch" and other[2] < 0:
            other = -other
        owner = drive["name"].removesuffix("_drive_package")

        def part(suffix, start, end, radius, kind):
            length = np.linalg.norm(end - start)
            mass = np.pi * radius**2 * length * 2700 if kind == "bracket" else 0.012
            result.append(
                dict(
                    name=f"{owner}_{suffix}",
                    body=drive["body"],
                    start=start.tolist(),
                    end=end.tolist(),
                    radius=radius,
                    owner=owner,
                    kind=kind,
                    mass_kg=float(mass),
                    provenance="custom aluminum saddle"
                    if kind == "bracket"
                    else "XT30(2+2) panel connector in custom saddle cradle; drive pigtail interface",
                )
            )

        # Two side rails with end ties, leaving both module faces accessible.
        spacing = drive["radius"] + 0.008
        front, rear = a - axis * 0.006, b + axis * 0.006
        for sign in (-1, 1):
            part(
                f"mount_rail_{sign}",
                front + radial * spacing * sign,
                rear + radial * spacing * sign,
                0.004,
                "bracket",
            )
        for tag, point in (("front", front), ("rear", rear)):
            # Radial ties are above the shaft; no fictitious solid disk through it.
            start = point + radial * spacing + other * spacing
            end = point - radial * spacing + other * spacing
            part(f"mount_tie_{tag}", start, end, 0.004, "bracket")
            for sign in (-1, 1):
                part(
                    f"mount_corner_{tag}_{sign}",
                    point + radial * spacing * sign,
                    point + radial * spacing * sign + other * spacing,
                    0.004,
                    "bracket",
                )
        if not electrical:
            continue
        if drive["shoulder_axis"]:
            from .shoulder_wiring import adapter_layout, connector_interfaces, fixed_adapter_path

            side, suffix = owner.split("_", 1)
            center, exit_axis = adapter_layout(suffix, 1 if side == "left" else -1)
            center, exit_axis = np.asarray(center), np.asarray(exit_axis)
            # Our keyed adapter/junction is mounted on the rear service tray.
            # Device-side socket clocking is provisional, explicitly separate.
            root = rear + other * spacing
            if suffix == "upper_roll":
                foot = center.copy()
                foot[1] = -0.04 if side == "left" else 0.04
                foot[2] = root[2]
                top = foot.copy()
                top[2] = center[2]
                for tag, first, last in (
                    ("foot", root, foot),
                    ("riser", foot, top),
                    ("tip", top, center),
                ):
                    part("connector_standoff_" + tag, first, last, 0.003, "bracket")
            elif suffix == "shoulder":
                foot = center.copy()
                foot[2] = root[2]
                part("connector_standoff_foot", root, foot, 0.003, "bracket")
                part("connector_standoff_tip", foot, center, 0.003, "bracket")
            else:
                part("connector_standoff", root, center, 0.003, "bracket")
            part(
                "power_can_connector",
                center - exit_axis * 0.01,
                center + exit_axis * 0.01,
                0.013,
                "connector",
            )
            result[-1]["box_size"] = [0.012, 0.020, 0.022]
            result[-1]["provenance"] = (
                "custom keyed power/CAN adapter; roll tray includes pass-through junction"
            )
            interfaces = connector_interfaces(suffix)
            for port_index, interface in enumerate(interfaces):
                lateral = (port_index - (len(interfaces) - 1) / 2) * (
                    0.028 if suffix == "shoulder" else 0.024
                )
                port = b + (
                    np.array([0.0, 0.0, lateral]) if suffix == "upper_roll" else radial * lateral
                )
                port_start, port_end = port + axis * 0.002, port + axis * 0.016
                radius = 0.007 if port_index == 0 else 0.004
                part(f"device_port_{port_index}", port_start, port_end, radius, "connector")
                result[-1].update(
                    interface=interface,
                    provenance="manufacturer connector family; rear-face clocking requires vendor CAD",
                )
                p0 = port_end
                p3 = (
                    center
                    - exit_axis * 0.01
                    + np.array(
                        [0.0, 0.0, lateral * 0.50]
                        if suffix == "shoulder"
                        else [0.0, lateral * 0.50, 0.0]
                    )
                )
                points = fixed_adapter_path(p0, p3, axis, exit_axis)
                for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
                    part(
                        f"drive_lead_{port_index}_{i}",
                        start,
                        end,
                        0.003 if port_index == 0 else 0.0015,
                        "wire",
                    )
                    result[-1].update(
                        mass_kg=0.005 / ((len(points) - 1) * len(interfaces)),
                        provenance="custom cut-to-length fixed adapter lead; not flex-rated stock lead",
                    )
            continue
        # Panel-side connector sits in a retaining cradle on the rear saddle tie.
        # This is our connector mount, not a guessed actuator PCB socket location.
        mount = rear + other * spacing
        part("connector_standoff", mount, mount + axis * 0.012, 0.004, "bracket")
        part("power_can_connector", mount + axis * 0.012, mount + axis * 0.032, 0.009, "connector")
        for sign in (-1, 1):
            part(
                f"connector_retainer_{sign}",
                mount + radial * sign * 0.012,
                mount + radial * sign * 0.012 + axis * 0.030,
                0.003,
                "bracket",
            )
        # Fixed four-conductor drive pigtail: rear adapter termination to panel.
        # This models our adapter interface, not unverified vendor cavity coordinates.
        p0 = b + axis * 0.004
        p3 = mount + axis * 0.032
        p1 = p0 + axis * 0.040
        p2 = p3 + axis * 0.040
        t = np.linspace(0.0, 1.0, 9)[:, None]
        points = (
            (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t**3 * p3
        )
        for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
            part(f"drive_lead_{i}", start, end, 0.0047, "wire")
            result[-1]["mass_kg"] = 0.005 / 8
            result[-1]["provenance"] = (
                "fixed adapter pigtail envelope; vendor 100 mm lead requires fit check"
            )
    return result


def assembly_capsules(model):
    return [
        Capsule(p["name"], p["body"], tuple(p["start"]), tuple(p["end"]), p["radius"])
        for p in assembly_parts(model) + harness_fixtures(model)
    ]


def harness_fixture_specs():
    """Bolted standoffs and split clamps at the dress-pack's explicit anchors.

    Custom geometry, shared with the renderer and motion checks. The root is on
    the host link; only the cable's short local attachment region is exempt.
    """
    from .harness import harness_spans

    fixtures = []
    for span in harness_spans():
        for endpoint, (name, point) in enumerate(zip(span["bodies"], span["anchors"])):
            normal = np.array(span["normals"][endpoint])
            anchor = np.array(point)
            root = anchor.copy()
            if name == "torso":
                root = np.array([-0.14, np.sign(anchor[1]) * 0.24, 0.20])
            else:
                root[1:] *= 0.4
            prefix = f"{span['owner']}_harness_{endpoint}"
            supports = [("support", root, anchor, 0.004, "bracket")]
            if name == "torso":
                rear = root.copy()
                rear[0] = -0.32
                outside = anchor.copy()
                outside[0] = -0.32
                supports = [
                    ("support_rear", root, rear, 0.004, "bracket"),
                    ("support_cross", rear, outside, 0.004, "bracket"),
                    ("support_tip", outside, anchor, 0.004, "bracket"),
                ]
            for tag, a, b, radius, kind in supports + [
                (
                    "clamp",
                    anchor - normal * 0.01,
                    anchor + normal * 0.01,
                    0.022,
                    "swivel_clamp" if span["owner"].endswith("_shoulder") else "clamp",
                )
            ]:
                fixtures.append(
                    dict(
                        name=f"{prefix}_{tag}",
                        owner=span["owner"],
                        body_name=name,
                        start=a.tolist(),
                        end=b.tolist(),
                        radius=radius,
                        kind=kind,
                        mass_kg=0.025
                        if kind in ("clamp", "swivel_clamp")
                        else float(np.pi * 0.004**2 * np.linalg.norm(b - a) * 2700),
                        provenance="custom split clamp and link-mounted aluminum standoff",
                    )
                )
    return fixtures


def harness_fixtures(model):
    return [dict(p, body=model.names.index(p["body_name"])) for p in harness_fixture_specs()]


def mechanical_assembly_parts(model):
    """Motor saddles only; detailed electrical packaging is deferred."""
    return assembly_parts(model, electrical=False)


def mechanical_assembly_capsules(model):
    return [
        Capsule(p["name"], p["body"], tuple(p["start"]), tuple(p["end"]), p["radius"])
        for p in mechanical_assembly_parts(model)
    ]
