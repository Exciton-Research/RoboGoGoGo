"""Constant-length external service loops for proximal joint harness analysis.

Kinematic centreline model, not cable elasticity/fatigue or electrical validation.
Uses conservative CFROBOT2.07.04.C diameter 8.5 mm (older manufacturer sheet;
newer sheet lists 8 mm), 10*d minimum bend and +/-180 degrees/m torsion.
A conservative 30 mm dress-pack envelope contains separately fused power branches
and a dedicated CFROBOT8.022 CAN cable; this is not connector pinout qualification.
"""

import numpy as np

CABLE_SOURCE = (
    "https://www.igus.it/contentData/Product_Files/Download/pdf/Data_sheet_chainflex_CFROBOT2.pdf"
)
RADIUS = 0.015
POWER_DIAMETER = 0.0085
CAN_DIAMETER = 0.0075
CAN_SOURCE = "https://us.misumi-ec.com/linked/pdf/vona/EU_Data_sheet_chainflex_CFROBOT8.pdf"
FREE_LENGTH = 0.70
MIN_BEND = 10 * POWER_DIAMETER + RADIUS + POWER_DIAMETER / 2
NEUTRAL = {"waist": 0.0, "upper_roll": 0.0, "shoulder": 1.2, "elbow": -1.35}


def loop_points(axis, delta, samples=49):
    axis = np.asarray(axis, float)
    radial = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.5 else np.array([0.0, 1, 0])
    radial -= axis * np.dot(axis, radial)
    radial /= np.linalg.norm(radial)
    turned = radial * np.cos(delta) + np.cross(axis, radial) * np.sin(delta)
    a, d = axis * 0.255 + radial * 0.21, axis * 0.255 - turned * 0.21
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
            # Route roll dress packs aft: a forward-facing arch would cross the
            # forearm when the arm extends along +X.
            sign = -1 if suffix == "upper_roll" else mounting_sign(side, suffix)
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
                    scope="48 V individual branches + separate 24 V wrist feed + CFROBOT8.022 CAN; connector pinout pending",
                    can_source=CAN_SOURCE,
                )
            )
    return result


def harness_clearance(model, q, pose, world, routes=None):
    """Sampled cable/rigid and inter-loop clearance, with chord padding.

    This validates the kinematic routing, not cable contact forces or fatigue.
    Same-loop neighbours are not collision pairs; each curve is a simple arch.
    """
    from ..planning.collision import Clearance, segment_distance

    routes = harness_routes(model, q, pose) if routes is None else routes
    for route in routes:
        if not route["valid"]:
            return Clearance(-1.0, (route["name"], "cable_bend_or_twist"))
    # 48 chords per loop; 2 mm inflation exceeds the cubic interpolation error
    # for this fixed-size arch (bulge <= .8 m, radius .2 m).
    starts, ends, groups = [], [], []
    for i, route in enumerate(routes):
        points = np.array(route["points"])
        starts.extend(points[:-1])
        ends.extend(points[1:])
        groups.extend([i] * (len(points) - 1))
    a, b, groups = np.array(starts), np.array(ends), np.array(groups)
    c, d = world.geometry(q)
    n, k = len(a), len(c)
    distances = segment_distance(
        np.repeat(a, k, axis=0), np.repeat(b, k, axis=0), np.tile(c, (n, 1)), np.tile(d, (n, 1))
    ).reshape(n, k)
    distances -= RADIUS + 0.002 + world.radii[None, :]
    row, col = np.unravel_index(np.argmin(distances), distances.shape)
    best = Clearance(
        float(distances[row, col]), (routes[groups[row]]["name"], world.capsules[col].name)
    )
    floor_distances = np.minimum(a[:, 2], b[:, 2]) - RADIUS - 0.002 - world.floor
    floor_index = int(np.argmin(floor_distances))
    if floor_distances[floor_index] < best.distance:
        best = Clearance(
            float(floor_distances[floor_index]),
            (routes[groups[floor_index]]["name"], "floor"),
        )
    i, j = np.triu_indices(n, 1)
    valid = groups[i] != groups[j]
    i, j = i[valid], j[valid]
    separation = segment_distance(a[i], b[i], a[j], b[j]) - 2 * (RADIUS + 0.002)
    index = int(np.argmin(separation))
    if separation[index] < best.distance:
        best = Clearance(
            float(separation[index]),
            (routes[groups[i[index]]]["name"], routes[groups[j[index]]]["name"]),
        )
    return best
