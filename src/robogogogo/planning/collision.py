"""Conservative capsule geometry for planning, independent of contact dynamics."""

import time
from dataclasses import dataclass

import jax
import numpy as np

from .. import spatial as sp


@dataclass(frozen=True)
class Capsule:
    name: str
    body: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class SphereObstacle:
    name: str
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class CapsuleObstacle:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float

    @property
    def center(self):
        return tuple((np.asarray(self.start) + self.end) / 2)


def segment_box_distance(a, b, center, size):
    """Exact segment/AABB distance by piecewise quadratic minimization."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lower, upper = (
        np.asarray(center) - np.asarray(size) / 2,
        np.asarray(center) + np.asarray(size) / 2,
    )
    direction = b - a
    cuts = [np.zeros(len(a)), np.ones(len(a))]
    for axis in range(3):
        for bound in (lower[axis], upper[axis]):
            cuts.append(
                np.clip(
                    np.divide(
                        bound - a[:, axis],
                        direction[:, axis],
                        out=np.zeros(len(a)),
                        where=np.abs(direction[:, axis]) > 1e-14,
                    ),
                    0,
                    1,
                )
            )
    cuts = np.sort(np.stack(cuts, axis=1), axis=1)
    mid = (cuts[:, :-1] + cuts[:, 1:]) / 2
    points = a[:, None, :] + mid[:, :, None] * direction[:, None, :]
    active = (points < lower) | (points > upper)
    boundary = np.where(points < lower, lower, upper)
    numerator = np.sum(direction[:, None, :] * (a[:, None, :] - boundary) * active, axis=-1)
    denominator = np.sum(direction[:, None, :] ** 2 * active, axis=-1)
    roots = np.clip(
        np.divide(-numerator, denominator, out=mid.copy(), where=denominator > 1e-14),
        cuts[:, :-1],
        cuts[:, 1:],
    )
    candidates = np.concatenate((cuts, roots), axis=1)
    points = a[:, None, :] + candidates[:, :, None] * direction[:, None, :]
    outside = np.maximum(np.maximum(lower - points, points - upper), 0)
    return np.sqrt(np.min(np.sum(outside**2, axis=-1), axis=1))


@dataclass(frozen=True)
class Clearance:
    distance: float
    pair: tuple[str, str]


def segment_distance(a, b, c, d):
    """Vectorized exact segment distance, including points and parallel segments."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = np.sum(u * u, axis=-1), np.sum(u * v, axis=-1), np.sum(v * v, axis=-1)
    dd, ee = np.sum(u * w, axis=-1), np.sum(v * w, axis=-1)
    denominator = aa * cc - bb * bb
    s = np.clip(
        np.divide(bb * ee - cc * dd, denominator, out=np.zeros_like(aa), where=denominator > 1e-14),
        0,
        1,
    )
    t = np.divide(bb * s + ee, cc, out=np.zeros_like(cc), where=cc > 1e-14)
    s = np.where(
        t < 0, np.clip(np.divide(-dd, aa, out=np.zeros_like(aa), where=aa > 1e-14), 0, 1), s
    )
    s = np.where(
        t > 1, np.clip(np.divide(bb - dd, aa, out=np.zeros_like(aa), where=aa > 1e-14), 0, 1), s
    )
    s = np.where(
        cc < 1e-14, np.clip(np.divide(-dd, aa, out=np.zeros_like(aa), where=aa > 1e-14), 0, 1), s
    )
    t = np.clip(t, 0, 1)
    # If first segment is a point, project onto the second.
    t = np.where(
        aa < 1e-14, np.clip(np.divide(ee, cc, out=np.zeros_like(cc), where=cc > 1e-14), 0, 1), t
    )
    return np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1)


def capsules_from_visuals(model, visuals):
    """Enclose primitive visual volumes. Cosmetic frontend shells are not included.

    Callers may supply measured Capsule geometry instead. This does not modify
    the dynamics engine or infer physical contact properties from visuals.
    """
    result = []
    for i, visual in enumerate(visuals):
        dims = np.asarray(visual["dimensions"], dtype=float)
        center = np.asarray(visual["origin"], dtype=float)
        rotation = np.asarray(sp.rpy_matrix(np.asarray(visual["rpy"])))
        shape = visual["shape"]
        axis = np.zeros(3)
        if shape == "box":
            longest = int(np.argmax(dims))
            axis[longest] = dims[longest] / 2
            radius = float(np.linalg.norm(np.delete(dims / 2, longest)))
        elif shape == "cylinder":
            radius = float(dims[0])
            axis[2] = dims[1] / 2
        elif shape == "sphere":
            radius = float(dims[0])
        else:
            raise ValueError(f"Unsupported collision proxy: {shape}")
        body = visual["body"]
        result.append(
            Capsule(
                f"{model.names[body]}:{i}",
                body,
                tuple(center - rotation @ axis),
                tuple(center + rotation @ axis),
                radius,
            )
        )
    return tuple(result)


class CollisionWorld:
    """Joint-space collision oracle with explicit adjacent-link exclusions.

    Edges are certified using a conservative kinematic displacement bound and
    adaptive subdivision, not just endpoint checks. Uncertified edges are rejected.
    The bound assumes revolute/prismatic trees with finite prismatic joint bounds.
    Continuous revolute coordinates are allowed; edges use their unwrapped values.
    """

    def __init__(
        self,
        model,
        capsules,
        *,
        obstacles=(),
        margin=0.015,
        floor=0.0,
        excluded_pairs=(),
        excluded_capsule_pairs=(),
        obstacle_exclusions=(),
        support_contacts=(),
    ):
        self.model, self.capsules, self.obstacles = model, tuple(capsules), tuple(obstacles)
        self.margin, self.floor = float(margin), float(floor)
        if not self.capsules or not np.isfinite([margin, floor]).all() or margin < 0:
            raise ValueError("Require collision geometry and finite nonnegative margin")
        self.lower, self.upper = np.asarray(model.lower), np.asarray(model.upper)
        prismatic = [
            model.indices[b]
            for b, kind in enumerate(model.kinds)
            if kind == "prismatic" and model.indices[b] >= 0
        ]
        if not np.isfinite([self.lower[prismatic], self.upper[prismatic]]).all():
            raise ValueError("Planner requires finite prismatic joint bounds")
        for c in self.capsules:
            if (
                not 0 <= c.body < len(model.names)
                or not np.isfinite([*c.start, *c.end, c.radius]).all()
                or len(c.start) != 3
                or len(c.end) != 3
                or c.radius <= 0
            ):
                raise ValueError("Invalid capsule geometry")
        for o in self.obstacles:
            if len(o.center) != 3 or not np.isfinite(o.center).all():
                raise ValueError("Invalid obstacle center")
            if isinstance(o, BoxObstacle):
                if len(o.size) != 3 or not np.isfinite(o.size).all() or min(o.size) <= 0:
                    raise ValueError("Invalid box size")
            elif not np.isfinite(o.radius) or o.radius <= 0:
                raise ValueError("Invalid obstacle radius")
        self.obstacle_exclusions = set(map(tuple, obstacle_exclusions))
        self.support_contacts = set(map(tuple, support_contacts))
        named_exclusions = {frozenset(pair) for pair in excluded_capsule_pairs}
        self._poses = jax.jit(lambda q: sp.forward_kinematics(model, q))
        self.bodies = np.array([c.body for c in self.capsules])
        self.starts = np.array([c.start for c in self.capsules])
        self.ends = np.array([c.end for c in self.capsules])
        self.radii = np.array([c.radius for c in self.capsules])
        ancestors = []
        for body in range(len(model.names)):
            chain = []
            while body >= 0:
                chain.append(body)
                body = model.parents[body]
            ancestors.append(chain)
        self.moving = np.array(
            [any(model.indices[b] >= 0 for b in ancestors[c.body]) for c in self.capsules]
        )
        excluded = {frozenset(pair) for pair in excluded_pairs}
        pairs = []
        for i, a in enumerate(self.capsules):
            for j, b in enumerate(self.capsules[:i]):
                if not (self.moving[i] or self.moving[j]) or a.body == b.body:
                    continue
                if frozenset((a.name, b.name)) in named_exclusions:
                    continue
                if frozenset((a.body, b.body)) in excluded:
                    continue
                pairs.append((i, j))
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        # Joint-specific bound on distance to descendant proxy endpoints.
        lengths = np.linalg.norm(np.asarray(model.origin), axis=1)
        self.weights = np.zeros(self.lower.shape, dtype=float)
        self.capsule_weights = np.zeros((len(self.capsules), len(self.lower)))
        influence = np.zeros_like(self.capsule_weights, dtype=bool)
        for row, capsule in enumerate(self.capsules):
            reach = max(np.linalg.norm(capsule.start), np.linalg.norm(capsule.end))
            body = capsule.body
            while body >= 0:
                index = model.indices[body]
                if index >= 0:
                    influence[row, index] = True
                    if model.kinds[body] == "prismatic":
                        self.capsule_weights[row, index] = 1.0
                        self.weights[index] = max(self.weights[index], 1.0)
                        reach += max(abs(self.lower[index]), abs(self.upper[index]))
                    else:
                        self.capsule_weights[row, index] = reach
                        self.weights[index] = max(self.weights[index], reach)
                reach += lengths[body]
                body = model.parents[body]
        i, j = self.pairs.T
        # A common ancestor rigidly transforms both capsules and cannot change
        # their separation. Only joints below their shared chain contribute.
        self.pair_weights = np.where(
            influence[i] & influence[j],
            0.0,
            self.capsule_weights[i] + self.capsule_weights[j],
        )
        self.checks = 0

    def geometry(self, q):
        pose = self._poses(q)
        rotations = np.asarray(pose.rotation)[self.bodies]
        positions = np.asarray(pose.position)[self.bodies]
        return (
            positions + np.einsum("nij,nj->ni", rotations, self.starts),
            positions + np.einsum("nij,nj->ni", rotations, self.ends),
        )

    def clearance(self, q, object_segment=None):
        self.checks += 1
        q = np.asarray(q, dtype=float)
        if (
            q.shape != self.lower.shape
            or not np.isfinite(q).all()
            or ((q < self.lower) | (q > self.upper)).any()
        ):
            return Clearance(float("-inf"), ("joint_limits", "robot"))
        radii = self.radii.copy()
        structural = object_segment is not None and np.shape(object_segment) == (5, 3)
        a, b = self.geometry(q)
        if object_segment is not None:
            a, b = a.copy(), b.copy()
            for index, capsule in enumerate(self.capsules):
                if capsule.name == "payload":
                    a[index], b[index] = object_segment[:2]
                    if structural:
                        radii[index] = object_segment[2, 0]
        best = Clearance(float("inf"), ("none", "none"))
        if len(self.pairs):
            i, j = self.pairs.T
            distances = segment_distance(a[i], b[i], a[j], b[j]) - radii[i] - radii[j]
            k = int(np.argmin(distances))
            best = Clearance(
                float(distances[k]), (self.capsules[i[k]].name, self.capsules[j[k]].name)
            )
        for obstacle in self.obstacles:
            if isinstance(obstacle, BoxObstacle):
                distances = segment_box_distance(a, b, obstacle.center, obstacle.size) - radii
            elif isinstance(obstacle, CapsuleObstacle):
                c, d = (
                    np.broadcast_to(
                        object_segment[0]
                        if object_segment is not None and obstacle.name == "transfer_bar"
                        else obstacle.start,
                        a.shape,
                    ),
                    np.broadcast_to(
                        object_segment[1]
                        if object_segment is not None and obstacle.name == "transfer_bar"
                        else obstacle.end,
                        a.shape,
                    ),
                )
                radius = (
                    object_segment[2, 0]
                    if structural and obstacle.name == "transfer_bar"
                    else obstacle.radius
                )
                distances = segment_distance(a, b, c, d) - radii - radius
            else:
                center = np.broadcast_to(obstacle.center, a.shape)
                distances = segment_distance(a, b, center, center) - radii - obstacle.radius
            for index, capsule in enumerate(self.capsules):
                pair = (capsule.name, obstacle.name)
                if pair in self.obstacle_exclusions:
                    distances[index] = np.inf
                elif pair in self.support_contacts:
                    # Intentional payload/support contact: at most 3 mm penetration,
                    # rather than the noncontact separation margin.
                    if (
                        structural
                        and capsule.name == "payload"
                        and isinstance(obstacle, BoxObstacle)
                        and np.all(
                            object_segment[4, :2]
                            >= np.asarray(obstacle.center[:2]) - np.asarray(obstacle.size[:2]) / 2
                        )
                        and np.all(
                            object_segment[3, :2]
                            <= np.asarray(obstacle.center[:2]) + np.asarray(obstacle.size[:2]) / 2
                        )
                    ):
                        distances[index] = object_segment[3, 2] - (
                            obstacle.center[2] + obstacle.size[2] / 2
                        )
                    distances[index] += self.margin + 0.003
            k = int(np.argmin(distances))
            if distances[k] < best.distance:
                best = Clearance(float(distances[k]), (self.capsules[k].name, obstacle.name))
        distances = np.where(self.moving, np.minimum(a[:, 2], b[:, 2]) - radii - self.floor, np.inf)
        if structural:
            for index, capsule in enumerate(self.capsules):
                if capsule.name == "payload":
                    distances[index] = object_segment[3, 2] - self.floor
        k = int(np.argmin(distances))
        if distances[k] < best.distance:
            best = Clearance(float(distances[k]), (self.capsules[k].name, "floor"))
        return best

    def edge_free(
        self, start, end, *, max_depth=14, object_start=None, object_end=None, deadline=None
    ):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        first = self.clearance(start, object_start)
        last = self.clearance(end, object_end)
        self.last_edge_clearance = min((first, last), key=lambda c: c.distance)
        ca, cb = first.distance, last.distance
        if min(ca, cb) < self.margin:
            return False

        if not self.obstacles and object_start is None and object_end is None:
            return self._self_edge_free(start, end, max_depth, deadline)

        def check(a, b, da, db, oa, ob, depth):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Collision edge certification exceeded the planning budget")
            bound = 2 * float(np.dot(self.weights, np.abs(b - a)))
            if oa is not None:
                bound += 2 * float(np.max(np.linalg.norm(ob - oa, axis=1)))
            if max(da, db) - bound >= self.margin + 1e-6:
                return True
            if depth == 0:
                return False
            mid = (a + b) / 2
            om = None if oa is None else (oa + ob) / 2
            clearance = self.clearance(mid, om)
            if clearance.distance < self.last_edge_clearance.distance:
                self.last_edge_clearance = clearance
            dm = clearance.distance
            return (
                dm >= self.margin
                and check(a, mid, da, dm, oa, om, depth - 1)
                and check(mid, b, dm, db, om, ob, depth - 1)
            )

        return check(start, end, ca, cb, object_start, object_end, max_depth)

    def _self_edge_free(self, start, end, max_depth, deadline):
        """Certify each self/floor pair with its own kinematic motion bound."""
        weights = np.concatenate((self.pair_weights, self.capsule_weights))

        def distances(q):
            self.checks += 1
            a, b = self.geometry(q)
            i, j = self.pairs.T
            pairs = segment_distance(a[i], b[i], a[j], b[j]) - self.radii[i] - self.radii[j]
            floor = np.where(
                self.moving, np.minimum(a[:, 2], b[:, 2]) - self.radii - self.floor, np.inf
            )
            values = np.r_[pairs, floor]
            index = int(np.argmin(values))
            if values[index] < self.last_edge_clearance.distance:
                names = (
                    tuple(self.capsules[k].name for k in self.pairs[index])
                    if index < len(self.pairs)
                    else (self.capsules[index - len(self.pairs)].name, "floor")
                )
                self.last_edge_clearance = Clearance(float(values[index]), names)
            return values

        def check(a, b, da, db, depth):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Collision edge certification exceeded the planning budget")
            bounds = np.einsum("ij,j->i", weights, np.abs(b - a))
            if np.all(np.maximum(da, db) - bounds >= self.margin + 1e-6):
                return True
            if depth == 0:
                return False
            mid = (a + b) / 2
            dm = distances(mid)
            return (
                np.min(dm) >= self.margin
                and check(a, mid, da, dm, depth - 1)
                and check(mid, b, dm, db, depth - 1)
            )

        return check(start, end, distances(start), distances(end), max_depth)


def adjacent_exclusions(model, *, hops=2):
    """Explicit structural exemptions for proxy volumes around attached joints."""
    pairs = []
    for body in range(len(model.names)):
        parent = body
        for _ in range(hops):
            parent = model.parents[parent]
            if parent < 0:
                break
            pairs.append((body, parent))
    return tuple(pairs)


class CollisionViolation(FloatingPointError):
    """Predicted dynamics rejected before the unsafe state was committed."""
