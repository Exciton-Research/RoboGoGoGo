"""Link-anchored shoulder, elbow and wrist dress packs for harness analysis.

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
MIN_BEND = 10 * POWER_DIAMETER + RADIUS + POWER_DIAMETER / 2


def harness_spans():
    """One dress-pack span per articulation group, not a loop per motor.

    All anchors are explicit local-frame fixtures. Motor supply leads are fixed
    on each motor's parent; only these spans cross moving link boundaries.
    """
    spans = []
    for side, sign in (("left", 1), ("right", -1)):
        for group, src, dst, a, d, length, joints, neutral in (
            (
                "shoulder",
                "torso",
                f"{side}_shoulder",
                [-0.28, sign * 0.58, 0.22],
                [0.23, sign * 0.14, 0.08],
                0.75,
                ("waist", "upper_roll", "shoulder"),
                (0.0, 0.0, 0.6),
            ),
            (
                "elbow",
                f"{side}_shoulder",
                f"{side}_elbow",
                [0.185, sign * 0.14, -0.06],
                [0.18, sign * 0.14, -0.06],
                0.50,
                ("elbow",),
                (-0.675,),
            ),
            (
                "wrist_pitch",
                f"{side}_elbow",
                f"{side}_tool",
                [0.24, sign * 0.12, 0.06],
                [0.0, sign * 0.12, 0.06],
                0.55,
                ("wrist_roll", "wrist_pitch", "tool_roll"),
                (0.0, 0.075, 0.0),
            ),
        ):
            spans.append(
                dict(
                    name=f"{side}_{group}_dress_pack",
                    owner=f"{side}_{group}",
                    bodies=(src, dst),
                    anchors=(a, d),
                    normal=[0, sign, 0],
                    normals=([0, sign, 0], [0, sign, 0]),
                    length_m=length,
                    joints=tuple(f"{side}_{j}" for j in joints),
                    neutral=neutral,
                )
            )
    return spans


def _routed_curve(a, d, u, v, length, samples=65):
    """Constant-length cubic with link-attached endpoint tangents.

    Reject impossible span lengths; never stretch a wire to make a pose feasible.
    """
    a, d, u, v = (np.asarray(x, float) for x in (a, d, u, v))
    t = np.linspace(0, 1, samples)[:, None]

    def curve(bulge):
        b, c = a + u * bulge, d + v * bulge
        return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t**3 * d

    low, high = 0.0, 1.0
    feasible = np.linalg.norm(d - a) <= length
    for _ in range(24):
        mid = (low + high) / 2
        if np.linalg.norm(np.diff(curve(mid), axis=0), axis=1).sum() < length:
            low = mid
        else:
            high = mid
    bulge = (low + high) / 2
    b, c = a + u * bulge, d + v * bulge
    velocity = 3 * (1 - t) ** 2 * (b - a) + 6 * (1 - t) * t * (c - b) + 3 * t * t * (d - c)
    acceleration = 6 * (1 - t) * (c - 2 * b + a) + 6 * t * (d - 2 * c + b)
    bend = float(
        np.min(
            np.linalg.norm(velocity, axis=1) ** 3
            / np.maximum(np.linalg.norm(np.cross(velocity, acceleration), axis=1), 1e-12)
        )
    )
    points = curve(bulge)
    actual = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    feasible = bool(feasible and abs(actual - length) < 1e-6)
    return points, bend, feasible


def routed_curve(a, d, u, v, length, samples=65):
    """Balance tangent lengths to avoid artificially pinching a skewed span.

    All candidates retain anchor tangents and physical length. This is a
    geometric routing model, not a cable elasticity/contact solver.
    """
    candidates = [
        _routed_curve(a, d, np.asarray(u) * ratio, np.asarray(v) / ratio, length, samples)
        for ratio in (0.60, 0.75, 0.90, 1.0, 1 / 0.90, 1 / 0.75, 1 / 0.60)
    ]
    return max(candidates, key=lambda item: (item[2], item[1]))


def harness_routes(model, q, pose):
    result = []
    for span in harness_spans():
        endpoints = []
        normals = []
        for endpoint, (name, anchor) in enumerate(zip(span["bodies"], span["anchors"])):
            body = model.names.index(name)
            rotation = np.asarray(pose.rotation[body])
            endpoints.append(np.asarray(pose.position[body]) + rotation @ anchor)
            normals.append(rotation @ span["normals"][endpoint])
        points, bend, feasible = routed_curve(*endpoints, *normals, span["length_m"])
        # Conservative total joint excursion, including wrist camera/tool leads.
        twist = (
            sum(
                abs(float(q[model.indices[model.names.index(j)]]) - neutral)
                for j, neutral in zip(span["joints"], span["neutral"])
            )
            / span["length_m"]
        )
        result.append(
            dict(
                name=span["name"],
                points=points.tolist(),
                radius=RADIUS,
                length_m=span["length_m"],
                min_bend_m=bend,
                bend_limit_m=MIN_BEND,
                twist_rad_m=twist,
                valid=feasible and bend >= MIN_BEND and twist <= np.pi,
                length_valid=feasible,
                source=CABLE_SOURCE,
                can_source=CAN_SOURCE,
                attachments=[f"{span['owner']}_harness_{i}" for i in (0, 1)],
                scope="grouped 48 V/24 V branches and CAN; link-mounted strain relief",
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
    # 64 chords per span; 2 mm padding covers cubic interpolation error for
    # the bounded 1 m tangent control lengths.
    starts, ends, groups, from_start, from_end = [], [], [], [], []
    for i, route in enumerate(routes):
        points = np.array(route["points"])
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        from_start.extend(arc[:-1])
        from_end.extend(arc[-1] - arc[1:])
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
    # Only the named clamp/standoff within 60 mm of its own cable end may touch.
    # The remainder of that cable still checks the same fixture.
    for group, route in enumerate(routes):
        for endpoint, prefix in enumerate(route.get("attachments", [])):
            arc = np.array(from_start if endpoint == 0 else from_end)
            attached = (groups == group) & (arc < 0.06)
            for col, capsule in enumerate(world.capsules):
                if capsule.name.startswith(prefix + "_"):
                    # Designed attachments use zero free-space margin beyond the
                    # clamped end, but penetration is still rejected there.
                    distances[groups == group, col] += world.margin
                    distances[attached, col] = np.inf
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
