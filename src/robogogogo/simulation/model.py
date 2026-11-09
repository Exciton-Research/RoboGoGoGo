"""Native JAX simulation model; right-handed, z-up, SI units."""

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .math import rpy_matrix


@dataclass(frozen=True)
class Model:
    parents: tuple[int, ...]  # -1 denotes the stationary world
    kinds: tuple[str, ...]
    indices: tuple[int, ...]  # -1 for a fixed joint
    names: tuple[str, ...]
    origin: jax.Array
    orientation: jax.Array
    axis: jax.Array
    mass: jax.Array
    com: jax.Array
    inertia: jax.Array  # about COM, expressed in the link frame
    damping: jax.Array
    effort: jax.Array
    lower: jax.Array
    upper: jax.Array
    home: jax.Array
    gravity: jax.Array
    dt: jax.Array
    sphere_body: jax.Array
    sphere_center: jax.Array
    sphere_radius: jax.Array
    contact_stiffness: jax.Array
    contact_damping: jax.Array
    friction: jax.Array
    limit_stiffness: jax.Array
    limit_damping: jax.Array


_METADATA = ("parents", "kinds", "indices", "names")


jax.tree_util.register_dataclass(
    Model,
    data_fields=[f.name for f in fields(Model) if f.name not in _METADATA],
    meta_fields=list(_METADATA),
)


class State(NamedTuple):
    q: jax.Array
    qd: jax.Array


class Pose(NamedTuple):
    position: jax.Array
    rotation: jax.Array


class Contacts(NamedTuple):
    position: jax.Array
    penetration: jax.Array
    force: jax.Array


class ModelBuilder:
    """Validated host-side construction of a topologically ordered robot tree.

    Fixed links may be massless. Each moving link must have positive mass and
    positive-definite, physically realizable COM inertia. Other geometry must
    be explicitly approximated by collision spheres for plane contacts.
    """

    def __init__(self, *, dt=0.002, gravity=(0.0, 0.0, -9.81)):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = dt
        self.gravity = self._vector(gravity, 3, "gravity")
        self.links = []
        self.spheres = []

    @staticmethod
    def _vector(value, size, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {size} finite values")
        return array

    def add_link(
        self,
        name,
        *,
        parent=-1,
        joint="revolute",
        origin=(0, 0, 0),
        rpy=(0, 0, 0),
        axis=(0, 0, 1),
        mass=1.0,
        com=(0, 0, 0),
        inertia=None,
        damping=0.1,
        effort=50.0,
        limits=(-math.inf, math.inf),
        home=0.0,
    ):
        if name in [link["name"] for link in self.links]:
            raise ValueError(f"duplicate link name: {name}")
        if not isinstance(parent, int) or parent < -1 or parent >= len(self.links):
            raise ValueError("parent must refer to an earlier link, or -1 for world")
        if joint not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"unsupported joint: {joint}")
        origin = self._vector(origin, 3, "origin")
        orientation = np.asarray(rpy_matrix(self._vector(rpy, 3, "rpy")))
        axis = self._vector(axis, 3, "axis")
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("joint axis must be nonzero")
        axis = axis / np.linalg.norm(axis)
        com = self._vector(com, 3, "com")
        if not math.isfinite(mass) or mass < 0 or (joint != "fixed" and mass == 0):
            raise ValueError("moving links require positive finite mass")
        if inertia is None:
            raise ValueError("COM inertia must be supplied explicitly")
        inertia = np.asarray(inertia, dtype=float)
        if (
            inertia.shape != (3, 3)
            or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T, atol=1e-10)
        ):
            raise ValueError("inertia must be a finite symmetric 3x3 matrix")
        eig = np.linalg.eigvalsh(inertia)
        if (
            (mass > 0 and eig[0] <= 0)
            or (mass == 0 and np.any(inertia != 0))
            or eig[-1] > eig[0] + eig[1] + 1e-8
        ):
            raise ValueError(
                "inertia must be positive definite and satisfy the triangle inequality (zero for a massless link)"
            )
        if not math.isfinite(damping) or damping < 0 or not math.isfinite(effort) or effort <= 0:
            raise ValueError("damping must be nonnegative and effort positive; both finite")
        low, high = limits
        if (
            math.isnan(low)
            or math.isnan(high)
            or low >= high
            or not math.isfinite(home)
            or not low <= home <= high
        ):
            raise ValueError("invalid limits or home outside limits")
        if joint == "continuous" and (math.isfinite(low) or math.isfinite(high)):
            raise ValueError("continuous joints cannot have position limits")
        self.links.append(
            dict(
                name=name,
                parent=parent,
                joint=joint,
                origin=origin,
                orientation=orientation,
                axis=axis,
                mass=mass,
                com=com,
                inertia=inertia,
                damping=damping,
                effort=effort,
                lower=low,
                upper=high,
                home=home,
            )
        )
        return len(self.links) - 1

    def add_sphere(self, body, center, radius):
        if not isinstance(body, int) or body < 0 or body >= len(self.links):
            raise ValueError("sphere body must be an existing link")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("sphere radius must be finite and positive")
        self.spheres.append((body, self._vector(center, 3, "sphere center"), radius))
        return self

    def build(
        self,
        *,
        contact_stiffness=3000.0,
        contact_damping=60.0,
        friction=0.6,
        limit_stiffness=500.0,
        limit_damping=20.0,
    ):
        values = (contact_stiffness, contact_damping, friction, limit_stiffness, limit_damping)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("contact/limit parameters must be finite and nonnegative")
        moving = [link for link in self.links if link["joint"] != "fixed"]
        if not moving:
            raise ValueError("a model needs at least one moving joint")
        indices, index = [], 0
        for link in self.links:
            indices.append(index if link["joint"] != "fixed" else -1)
            index += link["joint"] != "fixed"

        def array(key, links):
            return jnp.asarray(np.array([link[key] for link in links], dtype=float))

        return Model(
            tuple(link["parent"] for link in self.links),
            tuple(link["joint"] for link in self.links),
            tuple(indices),
            tuple(link["name"] for link in self.links),
            *(
                array(key, self.links)
                for key in ("origin", "orientation", "axis", "mass", "com", "inertia")
            ),
            *(array(key, moving) for key in ("damping", "effort", "lower", "upper", "home")),
            jnp.asarray(self.gravity),
            jnp.asarray(self.dt),
            jnp.asarray([s[0] for s in self.spheres], dtype=int),
            jnp.asarray(np.array([s[1] for s in self.spheres], dtype=float).reshape(-1, 3)),
            jnp.asarray([s[2] for s in self.spheres], dtype=float),
            *(jnp.asarray(v) for v in values),
        )
