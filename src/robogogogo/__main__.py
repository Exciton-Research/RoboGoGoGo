"""Launch the single-robot Studio or a guarded headless mobile run."""

import argparse
import json
from dataclasses import replace


def main():
    parser = argparse.ArgumentParser(description="RG Mobile Duo · robogogogo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if not args.headless:
        import uvicorn

        from .studio.api import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
        return
    from .robot.mobile_design import mobile_arm_definitions, mobile_dual_arm_urdf, mobile_home
    from .session import Session
    from .urdf import loads

    model = loads(mobile_dual_arm_urdf()).model
    model = replace(model, home=jnp.asarray(mobile_home(model)))
    session = Session(model, arms=mobile_arm_definitions(), mobile=True)
    for _ in range(args.steps):
        session.advance()
    result = dict(robot="RG Mobile Duo", seconds=session.elapsed, mobile=session.mobile_status)
    if args.export:
        result["trajectory"] = str(session.export(args.export))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
