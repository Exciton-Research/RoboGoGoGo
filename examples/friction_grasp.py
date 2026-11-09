"""Fixed-hand contact benchmark: close jaws, remove support, measure hold versus slip.

python examples/friction_grasp.py --friction 0.8 --force 40
python examples/friction_grasp.py --friction 0 --force 40 --output runs/frictionless.npz
This isolates finger/object mechanics. Studio additionally integrates arm reactions.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from robogogogo import spatial as sp
from robogogogo.manipulation import GraspObject
from robogogogo.simulation.grasp_contact import ContactState, contact_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--force", type=float, default=40.0)
    parser.add_argument("--output", type=Path, default=Path("runs/friction-grasp.npz"))
    args = parser.parse_args()
    if not 0 <= args.friction <= 2 or not 0 <= args.force <= 80:
        parser.error("Require friction in [0,2] and force in [0,80] N")
    jax.config.update("jax_enable_x64", True)
    obj = GraspObject()
    dt = 0.0002
    state = ContactState(
        jnp.array([0.2, 0.0, 0.5]),
        sp.rpy_matrix(jnp.array([0.0, 0.0, -np.pi / 2])),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.array([0.1]),
        jnp.zeros(1),
        jnp.zeros((1, 2, 6, 3)),
        jnp.zeros((6, 3)),
        jnp.array([0.1]),
    )

    @jax.jit
    def rollout(state, supports):
        state = state._replace(support_tangent=jnp.zeros((3 * (len(supports) + 1), 3)))

        def step(state, _):
            following, report = contact_step(
                state,
                jnp.array([[0.0, 0.0, 0.5]]),
                jnp.eye(3)[None],
                jnp.zeros((1, 3)),
                jnp.zeros((1, 3)),
                jnp.array([0.015]),
                jnp.array([args.force]),
                jnp.array([args.friction]),
                jnp.array([args.friction * 0.75]),
                obj.mass,
                jnp.asarray(obj.inertia),
                obj.radius,
                obj.half_length,
                jnp.array([0.0, 0.0, -9.81]),
                supports,
                dt,
            )
            return following, (
                following.position,
                following.width,
                report.normal,
                report.friction,
                report.slip_speed,
            )

        return jax.lax.scan(step, state, None, length=3750)

    closed, first = rollout(state, jnp.array([[0.2, 0.0, 0.25, 0.1, 0.47]]))
    final, second = rollout(closed, jnp.empty((0, 5)))
    arrays = [np.concatenate((a, b)) for a, b in zip(first, second)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=np.arange(1, len(arrays[0]) + 1) * dt,
        position=arrays[0],
        jaw_width=arrays[1],
        normal_force=arrays[2],
        friction_force=arrays[3],
        relative_speed=arrays[4],
        support_removed_at=0.75,
        mu_static=args.friction,
        mu_dynamic=0.75 * args.friction,
        force_limit_n=args.force,
    )
    print(
        json.dumps(
            dict(
                result="held" if float(final.position[2]) > 0.49 else "dropped",
                final_height_m=float(final.position[2]),
                normal_force_n=np.asarray(second[2][-1]).tolist(),
                output=str(args.output.resolve()),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
