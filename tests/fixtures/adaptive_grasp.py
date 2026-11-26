"""Internal numerical regression fixture; not a supported robot workflow."""

import jax.numpy as jnp
import numpy as np

from robogogogo import spatial as sp
from robogogogo.manipulation import Manipulation
from robogogogo.session import Session


def run(adaptive=True, friction=0.14, damage_force=55.0):
    builder = sp.ModelBuilder()
    parent = -1
    for i, axis in enumerate(np.eye(3)):
        parent = builder.add_link(
            f"joint_{i}",
            parent=parent,
            joint="prismatic",
            axis=axis,
            origin=(0, 0, 0.5) if i == 0 else (0, 0, 0),
            mass=1.0,
            inertia=np.eye(3) * 0.01,
            limits=(-1, 1),
            effort=100.0,
        )
    session = Session(builder.build(), arms=[dict(name="arm", tool="joint_2")])
    grasp = Manipulation(session.model, session.arms, [0.2, 0, 0.5], [])
    grasp.contact = grasp.contact._replace(
        rotation=sp.rpy_matrix(jnp.array([0, 0, -np.pi / 2])),
        width=jnp.array([0.0585]),
        drive_width=jnp.array([0.0545]),
    )
    grasp.configure_gripper("arm", force_limit_n=40.0 if adaptive else 8.0)
    if adaptive:
        grasp.enable_adaptive_grasp(initial_force_n=8, damage_force_n=damage_force)
    session.contact_system = grasp
    grasp.grasp("arm", session)
    for i in range(200):
        if i == 80:
            grasp.configure_gripper("arm", mu_static=friction, mu_dynamic=friction * 6 / 7)
        session.advance()
        if grasp.failure:
            break
    return dict(
        adaptive=adaptive,
        height_m=float(grasp.state.position[2]),
        failure=grasp.failure,
        feedback=grasp.summary()["adaptive_grasp"],
        damage_limit_exceeded=grasp.damage_limit_exceeded,
    )
