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
        return [dict(family="XT30 2P", terminals=["V+", "V-"]),
                dict(family="GH125 4P", terminals=["CAN_H", "CAN_L"],
                     unused_terminals="verify driver revision; no invented pin numbering")]
    return [dict(family="XT30 2+2", terminals=["V+", "V-", "CAN_H", "CAN_L"])]


def adapter_layout(suffix, sign):
    """Tray connector centre and cable exit axis in the drive's parent frame."""
    return {
        "waist": ([-0.025, 0, 0.214], [-1, 0, 0]),
        "upper_roll": ([-0.163, sign * 0.025, 0.025], [0, 0, 1]),
        "shoulder": ([-0.025, sign * 0.201, 0.025], [-1, 0, 0]),
    }[suffix]


def branch_specs(model):
    specs = []
    for side, sign in (("left", 1), ("right", -1)):
        # Separate ports on the roll tray: incoming from yaw, outgoing to pitch.
        for name, source, target, a, d, u, v, length in (
            ("yaw_roll", "waist", "upper_roll", [-0.035, 0, 0.214],
             [-0.163, sign * 0.025, 0.035], [-1, 0, 0], [0, 0, 1], 0.295),
            ("roll_pitch", "upper_roll", "shoulder", [-0.163, sign * 0.035, 0.025],
             [-0.035, sign * 0.201, 0.025], [0, sign, 0], [-1, 0, 0], 0.290),
        ):
            source_body = model.parents[model.names.index(f"{side}_{source}")]
            target_body = model.parents[model.names.index(f"{side}_{target}")]
            specs.append(dict(name=f"{side}_{name}_internal", side=side,
                              source=f"{side}_{source}_panel", target=f"{side}_{target}_panel",
                              bodies=(source_body, target_body), anchors=(a, d), normals=(u, v),
                              length_m=length, radius=0.008, bend_limit_m=0.040))
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
        twist = abs(float(q[model.indices[model.names.index(f"{spec['side']}_{joint}")]])) / spec["length_m"]
        routes.append(dict(**spec, points=points.tolist(), min_bend_m=bend,
                           twist_rad_m=twist, length_valid=feasible,
                           valid=feasible and bend >= spec["bend_limit_m"] and twist <= np.pi,
                           internal=True, guide_normals=[n.tolist() for n in normals],
                           swivel_angles_rad=[0., 0.], qualification="custom microloom requirements; supplier flex rating unverified"))
    return routes
