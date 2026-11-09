"""Native JAX simulation kinematics; right-handed, z-up, SI units."""

import jax
import jax.numpy as jnp

from .math import rotation
from .model import Pose


def forward_kinematics(model, q):
    positions, rotations = [], []
    for i, (parent, kind, index) in enumerate(zip(model.parents, model.kinds, model.indices)):
        p = jnp.zeros(3, dtype=q.dtype) if parent == -1 else positions[parent]
        r = jnp.eye(3, dtype=q.dtype) if parent == -1 else rotations[parent]
        joint_r = r @ model.orientation[i]
        point = p + r @ model.origin[i]
        if kind in ("revolute", "continuous"):
            joint_r = joint_r @ rotation(model.axis[i], q[index])
        elif kind == "prismatic":
            point = point + joint_r @ model.axis[i] * q[index]
        positions.append(point)
        rotations.append(joint_r)
    return Pose(jnp.stack(positions), jnp.stack(rotations))


def point_position(model, q, body=-1, local=(0.0, 0.0, 0.0)):
    pose = forward_kinematics(model, q)
    return pose.position[body] + pose.rotation[body] @ jnp.asarray(local)


def point_jacobian(model, q, body=-1, local=(0.0, 0.0, 0.0)):
    return jax.jacfwd(lambda x: point_position(model, x, body, local))(q)


def sphere_positions(model, q):
    pose = forward_kinematics(model, q)
    return pose.position[model.sphere_body] + jnp.einsum(
        "sij,sj->si", pose.rotation[model.sphere_body], model.sphere_center
    )
