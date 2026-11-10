"""Covered interconnects between the three shoulder drives.

Mount coordinates describe our adapter trays, not verified vendor PCB sockets.
The moving microloom specification is a design requirement awaiting supplier
qualification; manufacturer accessory leads are used only on fixed links.
"""

import numpy as np

CONNECTOR_SOURCE = "https://www.cubemars.com/product/cables.html"
SHOULDER_AXES = ("waist", "upper_roll", "shoulder")


def connector_interfaces(suffix):
    if suffix in ("upper_roll", "shoulder"):
        return [
            dict(family="XT30 2P", terminals=["V+", "V-"]),
            dict(
                family="GH125 4P",
                terminals=["CAN_H", "CAN_L"],
                unused_terminals="verify driver revision; no invented pin numbering",
            ),
        ]
    return [dict(family="XT30 2+2", terminals=["V+", "V-", "CAN_H", "CAN_L"])]


def adapter_layout(suffix, sign):
    """Tray connector centre and cable exit axis in the drive's parent frame."""
    return {
        "waist": ([-0.025, 0, 0.226], [-1, 0, 0]),
        "upper_roll": ([-0.163, sign * 0.025, 0.025], [0, 0, 1]),
        "shoulder": ([-0.025, sign * 0.201, sign * 0.025], [-1, 0, 0]),
    }[suffix]


def branch_specs(model):
    specs = []
    for side, sign in (("left", 1), ("right", -1)):
        # Separate ports on the roll tray: incoming from yaw, outgoing to pitch.
        for name, source, target, a, d, u, v, length in (
            (
                "yaw_roll",
                "waist",
                "upper_roll",
                [-0.035, 0, 0.226],
                [-0.163, sign * 0.025, 0.035],
                [-1, 0, 0],
                [0, 0, 1],
                0.295,
            ),
            (
                "roll_pitch",
                "upper_roll",
                "shoulder",
                [-0.163, sign * 0.035, 0.025],
                [-0.035, sign * 0.201, sign * 0.025],
                [0, sign, 0],
                [-1, 0, 0],
                0.250,
            ),
        ):
            source_body = model.parents[model.names.index(f"{side}_{source}")]
            target_body = model.parents[model.names.index(f"{side}_{target}")]
            specs.append(
                dict(
                    name=f"{side}_{name}_internal",
                    side=side,
                    source=f"{side}_{source}_panel",
                    target=f"{side}_{target}_panel",
                    bodies=(source_body, target_body),
                    anchors=(a, d),
                    normals=(u, v),
                    length_m=length,
                    radius=0.008,
                    bend_limit_m=0.040,
                )
            )
    return specs


def internal_routes(model, q, pose):
    from .harness import routed_curve

    routes = []
    for spec in branch_specs(model):
        endpoints, normals = [], []
        for body, anchor, normal in zip(spec["bodies"], spec["anchors"], spec["normals"]):
            rotation = np.asarray(pose.rotation[body])
            endpoints.append(np.asarray(pose.position[body]) + rotation @ anchor)
            normals.append(rotation @ normal)
        points, bend, feasible = routed_curve(*endpoints, *normals, spec["length_m"])
        joint = "waist" if "yaw_roll" in spec["name"] else "upper_roll"
        twist = (
            abs(float(q[model.indices[model.names.index(f"{spec['side']}_{joint}")]]))
            / spec["length_m"]
        )
        routes.append(
            dict(
                **spec,
                points=points.tolist(),
                min_bend_m=bend,
                twist_rad_m=twist,
                length_valid=feasible,
                valid=feasible and bend >= spec["bend_limit_m"] and twist <= np.pi,
                internal=True,
                guide_normals=[n.tolist() for n in normals],
                swivel_angles_rad=[0.0, 0.0],
                qualification="custom microloom requirements; supplier flex rating unverified",
            )
        )
    return routes


def cover_specs(model):
    """Fixed, removable service hoods, independent of the evaluated cable pose.

    Inner cavities are explicit capsule unions; 2 mm walls are represented in
    the outer envelopes. They are not regenerated to follow a failed cable.
    """
    from .packaging import shoulder_packages

    covers = []
    for side, sign in (("left", 1), ("right", -1)):
        parent = model.parents[model.names.index(f"{side}_waist")]
        yaw = model.names.index(f"{side}_waist")
        paths = [
            (
                "yaw_transfer",
                parent,
                [[-0.025, 0, 0.226], [-0.13, 0, 0.20], [-0.17, 0, 0.06]],
                0.080,
            ),
            (
                "roll_transfer",
                yaw,
                [
                    [-0.17, sign * 0.03, 0.055 if sign > 0 else 0.025],
                    [-0.15, sign * 0.15, 0.065 if sign > 0 else 0.025],
                    [-0.035, sign * 0.20, 0.065 if sign > 0 else 0.025],
                ],
                0.055,
            ),
        ]
        for name, body, points, radius in paths:
            for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
                covers.append(
                    dict(
                        name=f"{side}_{name}_hood_{i}",
                        side=side,
                        body=body,
                        start=a,
                        end=b,
                        inner_radius=radius,
                        radius=radius + 0.002,
                    )
                )
    for package in shoulder_packages(model):
        if not package["shoulder_axis"]:
            continue
        a, b = np.asarray(package["start"]), np.asarray(package["end"])
        axis = (b - a) / np.linalg.norm(b - a)
        covers.append(
            dict(
                name=package["name"].replace("drive_package", "service_hood"),
                side=package["name"].split("_")[0],
                body=package["body"],
                start=(b + axis * 0.005).tolist(),
                end=(b + axis * 0.015).tolist(),
                inner_radius=0.056 if package["shoulder_axis"] == "pitch" else 0.060,
                radius=0.058 if package["shoulder_axis"] == "pitch" else 0.062,
            )
        )
    return covers


def _cylinder_distance(points, start, end, radius):
    """Signed point distance to a finite, solid actuator cylinder."""
    axis = end - start
    length = np.linalg.norm(axis)
    axis /= length
    delta = points - (start + end) / 2
    z = delta @ axis
    radial = np.linalg.norm(delta - z[:, None] * axis, axis=1) - radius
    axial = np.abs(z) - length / 2
    return np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(
        np.maximum(radial, axial), 0
    )


def internal_clearance(model, q, pose, world, routes=None):
    """Guard-equivalent clearance; 1 mm internal installation gap.

    Interior cables check solid actuator cylinders, not enclosing motor-cover
    spheres. Cover walls are checked by containment in fixed inner cavities.
    Every chord uses half its length as a conservative point-sampling bound.
    Only 20 mm at a named adapter termination may contact that adapter.
    """
    from ..planning.collision import Clearance, segment_distance
    from .packaging import shoulder_packages

    routes = internal_routes(model, q, pose) if routes is None else routes
    covers = cover_specs(model)
    packages = shoulder_packages(model)
    best = Clearance(float("inf"), ("", ""))
    c, d = world.geometry(q)

    def transformed(body, value):
        return np.asarray(pose.position[body]) + np.asarray(pose.rotation[body]) @ value

    def keep(distance, pair):
        nonlocal best
        report = Clearance(float(distance) + world.margin - 0.001, pair)
        if report.distance < best.distance:
            best = report

    for route in routes:
        if not route["valid"]:
            return Clearance(-1.0, (route["name"], "internal_bend_twist_or_length"))
        points = np.asarray(route["points"])
        mid = (points[:-1] + points[1:]) / 2
        bound = np.linalg.norm(np.diff(points, axis=0), axis=1) / 2
        radius = route["radius"]
        # Each ball enclosing a chord must fit within a real cavity. Covers are
        # fixed to link frames, never fitted to the current cable shape.
        containment = np.full(len(mid), -np.inf)
        for cover in covers:
            if cover["side"] != route["side"]:
                continue
            a = transformed(cover["body"], cover["start"])
            b = transformed(cover["body"], cover["end"])
            distance = segment_distance(
                mid, mid, np.broadcast_to(a, mid.shape), np.broadcast_to(b, mid.shape)
            )
            containment = np.maximum(containment, cover["inner_radius"] - distance - radius - bound)
        keep(np.min(containment), (route["name"], "cover_inner_wall"))
        for package in packages:
            a = transformed(package["body"], package["start"])
            b = transformed(package["body"], package["end"])
            gap = _cylinder_distance(mid, a, b, package["radius"]) - radius - bound
            keep(np.min(gap), (route["name"], package["name"]))
        gaps = (
            segment_distance(
                np.repeat(points[:-1], len(c), axis=0),
                np.repeat(points[1:], len(c), axis=0),
                np.tile(c, (len(mid), 1)),
                np.tile(d, (len(mid), 1)),
            ).reshape(len(mid), len(c))
            - radius
            - world.radii
        )
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        for col, capsule in enumerate(world.capsules):
            if (
                capsule.name.endswith("_drive_package")
                or capsule.name == f"{route['side']}_socket"
                or "_hood" in capsule.name
            ):
                gaps[:, col] = np.inf
            for end, key in ((0, "source"), (1, "target")):
                owner = route[key].removesuffix("_panel")
                if capsule.name.startswith(owner + "_") and any(
                    tag in capsule.name for tag in ("connector", "drive_lead")
                ):
                    local = arc[:-1] < 0.020 if end == 0 else arc[-1] - arc[1:] < 0.020
                    gaps[local, col] = np.inf
        row, col = np.unravel_index(np.argmin(gaps), gaps.shape)
        keep(gaps[row, col], (route["name"], world.capsules[col].name))
        keep(np.min(points[:, 2]) - radius - world.floor, (route["name"], "floor"))
    # Moving branch pairs may meet only at their declared common roll junction.
    for i, first in enumerate(routes):
        for second in routes[i + 1 :]:
            a, b = np.asarray(first["points"]), np.asarray(second["points"])
            gaps = segment_distance(
                np.repeat(a[:-1], len(b) - 1, axis=0),
                np.repeat(a[1:], len(b) - 1, axis=0),
                np.tile(b[:-1], (len(a) - 1, 1)),
                np.tile(b[1:], (len(a) - 1, 1)),
            ).reshape(len(a) - 1, len(b) - 1)
            gaps -= first["radius"] + second["radius"]
            if first["target"] == second["source"]:
                aa = np.linalg.norm((a[:-1] + a[1:]) / 2 - a[-1], axis=1) < 0.025
                bb = np.linalg.norm((b[:-1] + b[1:]) / 2 - b[0], axis=1) < 0.025
                gaps[np.ix_(aa, bb)] = np.inf
            keep(np.min(gaps), (first["name"], second["name"]))
    return best


def fixed_adapter_path(start, end, source_axis, target_axis, radius=0.010):
    """Two filleted bends in a rear service bay, without cubic tangent cusps."""
    start, end, source_axis, target_axis = map(np.asarray, (start, end, source_axis, target_axis))
    corners = [start, start + source_axis * 0.050, end - target_axis * 0.050, end]
    transitions = []
    for a, b, c in zip(corners, corners[1:], corners[2:]):
        incoming, outgoing = b - a, c - b
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        angle = np.arccos(np.clip(np.dot(incoming, outgoing), -1, 1))
        distance = radius * np.tan(angle / 2)
        before, after = b - incoming * distance, b + outgoing * distance
        normal = np.cross(incoming, outgoing)
        normal /= np.linalg.norm(normal)
        center = before + np.cross(normal, incoming) * radius
        radial = before - center
        ts = np.linspace(0, angle, 13)
        arc = center + np.cos(ts)[:, None] * radial + np.sin(ts)[:, None] * np.cross(normal, radial)
        transitions.append((distance, arc))
    if transitions[0][0] + transitions[1][0] >= np.linalg.norm(corners[2] - corners[1]):
        raise ValueError("Fixed lead bends cannot fit between tray and device")
    return np.vstack((start, transitions[0][1], transitions[1][1], end))
