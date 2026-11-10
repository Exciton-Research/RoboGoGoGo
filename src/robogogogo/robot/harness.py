"""Constant-length external service loops for proximal joint harness analysis.

Kinematic centreline model, not cable elasticity/fatigue or electrical validation.
Uses conservative CFROBOT2.07.04.C diameter 8.5 mm (older manufacturer sheet;
newer sheet lists 8 mm), 10*d minimum bend and +/-180 degrees/m torsion.
Power branch selection does not specify the separate impedance-controlled CAN pair.
"""

import numpy as np

CABLE_SOURCE = (
    "https://www.igus.it/contentData/Product_Files/Download/pdf/Data_sheet_chainflex_CFROBOT2.pdf"
)
RADIUS = 0.00425
FREE_LENGTH = 0.65
MIN_BEND = 10 * 2 * RADIUS + RADIUS
NEUTRAL = {"waist": 0.0, "upper_roll": 0.0, "shoulder": 1.2, "elbow": -1.35}


def loop_points(axis, delta, samples=49):
    axis = np.asarray(axis, float)
    radial = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.5 else np.array([0.0, 1, 0])
    radial -= axis * np.dot(axis, radial)
    radial /= np.linalg.norm(radial)
    turned = radial * np.cos(delta) + np.cross(axis, radial) * np.sin(delta)
    a, d = axis * 0.215 + radial * 0.18, axis * 0.215 - turned * 0.18
    t = np.linspace(0, 1, samples)[:, None]

    def curve(bulge):
        b, c = a + axis * bulge, d + axis * bulge
        return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t**3 * d

    low, high = 0.10, 0.8
    for _ in range(22):
        middle = (low + high) / 2
        length = np.linalg.norm(np.diff(curve(middle), axis=0), axis=1).sum()
        if length < FREE_LENGTH:
            low = middle
        else:
            high = middle
    bulge = (low + high) / 2
    b, c = a + axis * bulge, d + axis * bulge
    velocity = 3 * (1 - t) ** 2 * (b - a) + 6 * (1 - t) * t * (c - b) + 3 * t * t * (d - c)
    acceleration = 6 * (1 - t) * (c - 2 * b + a) + 6 * t * (d - 2 * c + b)
    bend = np.min(
        np.linalg.norm(velocity, axis=1) ** 3
        / np.maximum(np.linalg.norm(np.cross(velocity, acceleration), axis=1), 1e-12)
    )
    return curve(bulge), float(bend)


def harness_routes(model, q, pose):
    from .components import mounting_sign

    result = []
    for side in ("left", "right"):
        for suffix, neutral in NEUTRAL.items():
            body = model.names.index(f"{side}_{suffix}")
            parent, index = model.parents[body], model.indices[body]
            sign = mounting_sign(side, suffix)
            delta = float(q[index]) - neutral
            points, bend = loop_points(np.asarray(model.axis[body]) * sign, delta * sign)
            rotation = np.asarray(pose.rotation[parent]) @ np.asarray(model.orientation[body])
            origin = np.asarray(pose.position[parent]) + np.asarray(
                pose.rotation[parent]
            ) @ np.asarray(model.origin[body])
            points = points @ rotation.T + origin
            result.append(
                dict(
                    name=f"{side}_{suffix}_service_loop",
                    points=points.tolist(),
                    radius=RADIUS,
                    length_m=FREE_LENGTH,
                    min_bend_m=bend,
                    bend_limit_m=MIN_BEND,
                    twist_rad_m=abs(delta) / FREE_LENGTH,
                    valid=bend >= MIN_BEND and abs(delta) / FREE_LENGTH <= np.pi,
                    source=CABLE_SOURCE,
                    scope="power branch mechanical routing; CAN branch pending",
                )
            )
    return result
