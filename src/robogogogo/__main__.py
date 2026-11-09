"""Launch the Python 3D workbench or a reproducible headless experiment."""

import argparse
import json

import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="robogogogo native Python/JAX robotics simulator")
    parser.add_argument("--scene", choices=("arm", "drop"), default="arm")
    parser.add_argument("--urdf", help="Load the supported fixed-base URDF subset")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=300, help="Control ticks in headless mode")
    parser.add_argument("--dt", type=float, default=0.002, help="Physics timestep in seconds")
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--export", help="NPZ trajectory output path")
    args = parser.parse_args()
    if args.steps < 1 or args.substeps < 1:
        parser.error("steps and substeps must be positive")
    if args.urdf and args.scene != "arm":
        parser.error("--urdf requires --scene arm")
    # Explicit application-level choice; importing the library changes no JAX settings.
    jax.config.update("jax_enable_x64", True)
    from . import models, urdf
    from .spatial import plane_contacts, point_position
    from .viewer import Session, Workbench

    try:
        model = (
            urdf.load(args.urdf, dt=args.dt).model
            if args.urdf
            else models.arm6(dt=args.dt)
            if args.scene == "arm"
            else models.slider(dt=args.dt)
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("Compiling native JAX dynamics…", flush=True)
    session = Session(model, scene=args.scene, substeps=args.substeps)
    if args.headless:
        if args.scene == "arm":
            delta = np.zeros_like(model.home)
            delta[: min(3, delta.size)] = [0.3, 0.15, -0.2][: min(3, delta.size)]
            session.set_joints(model.home + delta)
        for _ in range(args.steps):
            session.advance()
        result = dict(
            scene=args.scene,
            steps=args.steps,
            simulation_seconds=session.elapsed,
            q=np.asarray(session.state.q).tolist(),
        )
        if args.scene == "arm":
            result["tip_error_m"] = float(
                np.linalg.norm(np.asarray(point_position(model, session.state.q)) - session.target)
            )
        else:
            contacts, _ = plane_contacts(model, session.state)
            result["ground_penetration_m"] = float(contacts.penetration[0])
            result["normal_force_N"] = float(contacts.force[0, 2])
        if args.export:
            result["trajectory"] = str(session.export(args.export))
        print(json.dumps(result, indent=2))
    else:
        Workbench(session, export_path=args.export).show()


if __name__ == "__main__":
    main()
