"""Native articulated material grasp benchmark; no renderer or external engine.

python examples/material_grasp.py --material soft_foam --force 4
python examples/material_grasp.py --material fragile_shell --force 40 --no-adaptive
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from robogogogo import spatial as sp
from robogogogo.manipulation import Manipulation
from robogogogo.session import Session
from robogogogo.simulation.materials import MATERIALS


def run(material="soft_foam", force=None, adaptive=True, duration=2.0, force_threshold=None):
    selected = MATERIALS[material]
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
    grasp = Manipulation(
        session.model, session.arms, [0.2, 0, 0.5], [[0.2, 0, 0.15, 0.12, 0.47]], material=material
    )
    grasp.contact = grasp.contact._replace(
        rotation=sp.rpy_matrix(jnp.array([0.0, 0.0, -np.pi / 2]))
    )
    initial_force = force if force is not None else selected.initial_force_n
    grasp.configure_gripper("arm", force_limit_n=40.0 if adaptive else initial_force)
    if adaptive:
        grasp.enable_adaptive_grasp(
            initial_force_n=initial_force,
            damage_force_n=force_threshold
            if force_threshold is not None
            else selected.force_threshold_n,
        )
    session.contact_system = grasp
    grasp.grasp("arm", session)
    secured_at = None
    lift_height = 0.0
    for _ in range(round(duration / 0.01)):
        if grasp.owner is not None and secured_at is None:
            secured_at = session.elapsed
        if secured_at is not None and session.elapsed - secured_at > 0.2 and not grasp.motion_hold:
            lift_height = min(0.04, lift_height + 0.0004)
            session.target_q = jnp.array([0.0, 0.0, lift_height])
        session.advance()
        if grasp.failure:
            break
    result = grasp.summary()
    return dict(
        time_s=session.elapsed,
        failure=grasp.failure,
        object=result["object"],
        grippers=result["grippers"],
        feedback=result["adaptive_grasp"],
        events=result["events"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=MATERIALS, default="soft_foam")
    parser.add_argument("--force", type=float)
    parser.add_argument("--no-adaptive", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = run(args.material, args.force, not args.no_adaptive)
    for key in ("indentation_m", "permanent_m", "damage"):
        result["object"]["material"].pop(key, None)
    print(json.dumps(result, indent=2))
