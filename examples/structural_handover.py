"""Native articulated two-arm structural grasp and guarded load-transfer benchmark."""

import argparse
import json

import jax
import numpy as np

from robogogogo import spatial as sp
from robogogogo.manipulation import Manipulation
from robogogogo.session import Session


def make_rig(material="soft_foam", force=4.0, adaptive=True):
    builder = sp.ModelBuilder()
    for arm, y in (("left", -0.09), ("right", 0.09)):
        parent = -1
        for i, axis in enumerate(((0, 1, 0), (-1, 0, 0), (0, 0, 1))):
            parent = builder.add_link(
                f"{arm}_{i}",
                parent=parent,
                joint="prismatic",
                axis=axis,
                origin=(0.2, y + 0.2, 0.5) if i == 0 else (0, 0, 0),
                rpy=(0, 0, -np.pi / 2) if i == 0 else (0, 0, 0),
                mass=1,
                inertia=np.eye(3) * 0.01,
                limits=(-1, 1),
                effort=100,
            )
    session = Session(
        builder.build(), arms=[dict(name=a, tool=f"{a}_2") for a in ("left", "right")]
    )
    manipulation = Manipulation(
        session.model,
        session.arms,
        [0.2, 0, 0.5],
        [[0.2, 0, 0.1, 0.02, 0.48]],
        material=material,
        structural=True,
    )
    for name in manipulation.names:
        manipulation.configure_gripper(name, force_limit_n=40.0 if adaptive else force)
    if adaptive:
        manipulation.enable_adaptive_grasp(
            initial_force_n=force, damage_force_n=max(40.0, force / 0.75)
        )
    session.contact_system = manipulation
    manipulation.grasp("left", session)
    return session, manipulation


def run(
    material="soft_elastomer", force=16.0, lift=0.1, duration=12.0, transfer=True, adaptive=True
):
    session, grasp = make_rig(material, force, adaptive)
    height, ready, maximum_risk, maximum_bend = 0.0, None, 0.0, 0.0
    for _ in range(round(duration / 0.01)):
        if grasp.owner == "left" and not grasp.motion_hold and grasp.pending is None:
            height = min(lift, height + 0.0004)
            session.target_q = session.target_q.at[2].set(height).at[5].set(height)
            if height >= lift and ready is None:
                ready = session.elapsed
        if (
            transfer
            and ready is not None
            and session.elapsed > ready + 0.3
            and grasp.owner == "left"
            and not grasp.pending
        ):
            grasp.transfer("right", session)
        session.advance()
        maximum_risk = max(maximum_risk, grasp.material_status()["risk"])
        maximum_bend = max(
            maximum_bend, grasp.structural.summary(grasp.contact.structure)["bend_m"]
        )
        if grasp.failure:
            break
    return dict(
        time_s=session.elapsed,
        maximum_risk=maximum_risk,
        maximum_bend_m=maximum_bend,
        **grasp.summary(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material",
        default="soft_elastomer",
        choices=["soft_foam", "soft_elastomer", "fragile_shell"],
    )
    parser.add_argument("--force", type=float, default=16.0)
    parser.add_argument("--lift", type=float, default=0.1)
    parser.add_argument("--no-adaptive", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = run(args.material, args.force, args.lift, adaptive=not args.no_adaptive)
    print(json.dumps(result, indent=2))
