"""Run the dual-arm task without a browser, FastAPI, OpenCV, or external physics.

python examples/coordinated_inspection.py --output runs/coordinated-inspection.npz
The trajectory contains physics states; use Studio for synchronized camera datasets.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from robogogogo import models, urdf
from robogogogo.contracts import Observation
from robogogogo.planning import (
    CollisionWorld,
    SphereObstacle,
    TaskExecutor,
    adjacent_exclusions,
    capsules_from_visuals,
    plan_shared_inspection,
)
from robogogogo.session import Session
from robogogogo.system import ControlSystem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/coordinated-inspection.npz")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    robot = urdf.loads(models.multi_arm_urdf(2))
    model = replace(robot.model, home=jnp.tile(models.arm6().home, 2))
    definitions = [
        dict(name="left_arm", tool="left_tool"),
        dict(name="right_arm", tool="right_tool"),
    ]
    session = Session(model, arms=definitions)
    world = CollisionWorld(
        model,
        capsules_from_visuals(model, robot.visuals),
        excluded_pairs=adjacent_exclusions(model),
        obstacles=[SphereObstacle("inspection_fixture", (0.55, 0, 0.45), 0.1)],
    )
    plan = plan_shared_inspection(model, session.arms, world, model.home)
    executor = TaskExecutor(plan, max_delta=np.full(model.home.size, 0.08))
    system = ControlSystem(
        {"task": executor},
        lower=model.lower,
        upper=model.upper,
        max_delta=executor.limit,
        collision_world=world,
    )
    minimum = float("inf")
    for index in range(250):
        session.targets = {k: np.asarray(v) for k, v in executor.stage.targets.items()}
        observation = Observation(
            "Inspect the shared station.",
            b"",
            tuple(np.asarray(session.state.q)),
            tuple(np.asarray(session.state.qd)),
            session.elapsed,
        )
        decision = system.step(observation, controller="task")
        session.target_q = session.state.q + jnp.asarray(decision.action.actions[0])
        for _ in range(decision.action.control_ticks):
            session.advance(collision_world=world)
            minimum = min(minimum, world.clearance(session.state.q).distance)
        pose = session.poses()
        executor.update(
            np.asarray(session.state.q),
            np.asarray(session.state.qd),
            session.elapsed,
            {arm.name: np.asarray(pose.position[arm.body]) for arm in session.arms},
        )
        if executor.status in ("completed", "blocked"):
            break
    if executor.status != "completed":
        raise RuntimeError(f"Task did not finish: {executor.summary()}")
    destination = session.export(args.output)
    report = dict(
        actions=index + 1,
        simulation_seconds=session.elapsed,
        minimum_clearance_m=minimum,
        plan=plan.to_dict(),
        result=executor.summary(),
    )
    Path(destination).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(
        f"Completed all {len(plan.stages)} stages in {index + 1} actions; minimum sampled clearance {minimum * 1000:.1f} mm."
    )
    print(destination)


if __name__ == "__main__":
    main()
