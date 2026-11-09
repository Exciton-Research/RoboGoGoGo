"""Native JAX simulation math; right-handed, z-up, SI units."""

import jax.numpy as jnp


def skew(v):
    x, y, z = v
    zero = jnp.zeros_like(x)
    return jnp.array([[zero, -z, y], [z, zero, -x], [-y, x, zero]])


def rotation(axis, angle):
    """Rodrigues rotation; axis is a unit vector validated by the builder."""
    cross = skew(axis)
    return (
        jnp.eye(3, dtype=axis.dtype)
        + jnp.sin(angle) * cross
        + (1 - jnp.cos(angle)) * (cross @ cross)
    )


def rpy_matrix(rpy):
    """URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    return (
        rotation(jnp.array([0.0, 0.0, 1.0]), y)
        @ rotation(jnp.array([0.0, 1.0, 0.0]), p)
        @ rotation(jnp.array([1.0, 0.0, 0.0]), r)
    )
