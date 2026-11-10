"""Reproducible nominal-range audit; samples are not a continuous-space certificate.

python -m robogogogo.robot.validate_harness --output /tmp/harness-audit.json
Returns exit code 2 when a sampled configuration is unsafe.
"""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from ..urdf import loads
from .definition import ROBOT_REVISION
from .harness import harness_clearance, harness_routes
from .mobile_design import mobile_dual_arm_urdf, mobile_home
from .self_collision import mobile_collision_world


def configurations(model, samples=29, random_samples=128):
    arms = [
        i
        for i, n in enumerate(model.names)
        if n.startswith(("left_", "right_")) and model.indices[i] >= 0
    ]
    torso = [model.names.index(n) for n in ("torso_yaw", "torso_lower", "torso_upper", "torso")]
    for posture in ("standing", "low", "reach"):
        home = mobile_home(model, posture)
        yield f"{posture}:home", home.copy()
        for body in arms + torso:
            index = model.indices[body]
            for value in np.linspace(float(model.lower[index]), float(model.upper[index]), samples):
                q = home.copy()
                q[index] = value
                yield f"{posture}:{model.names[body]}={value:.6f}", q
        # Coupled wrist corners and midpoint, not just independent-axis sweeps.
        for values in product((-1.0, 0.0, 1.0), repeat=3):
            q = home.copy()
            for side in ("left", "right"):
                for suffix, value in zip(("wrist_roll", "wrist_pitch", "tool_roll"), values):
                    index = model.indices[model.names.index(f"{side}_{suffix}")]
                    q[index] = (
                        float(model.lower[index])
                        if value < 0
                        else float(model.upper[index])
                        if value > 0
                        else 0.0
                    )
            yield f"{posture}:coupled-wrist:{values}", q
    rng = np.random.default_rng(20261110)
    for trial in range(random_samples):
        q = mobile_home(model, ("standing", "low", "reach")[trial % 3])
        for body in arms + torso:
            index = model.indices[body]
            q[index] = rng.uniform(float(model.lower[index]), float(model.upper[index]))
        yield f"coupled-random:{trial}", q


def audit(samples=29, random_samples=128):
    if samples < 3 or random_samples < 0:
        raise ValueError("Use at least three axis samples and nonnegative random samples")
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    counts = Counter()
    pairs = Counter()
    examples = []
    worst = None
    for label, q in configurations(model, samples, random_samples):
        rigid = world.clearance(q)
        pose = world._poses(q)
        routes = harness_routes(model, q, pose)
        cable = harness_clearance(model, q, pose, world, routes)
        counts["configurations"] += 1
        bad_rigid = rigid.distance < world.margin
        bad_cable = cable.distance < world.margin
        counts["rigid_failures"] += int(bad_rigid)
        counts["harness_failures"] += int(bad_cable)
        counts["clear"] += int(not bad_rigid and not bad_cable)
        for report, bad in ((rigid, bad_rigid), (cable, bad_cable)):
            if bad:
                pairs[" / ".join(report.pair)] += 1
        if bad_rigid or bad_cable:
            report = min((rigid, cable), key=lambda x: x.distance)
            record = dict(
                case=label,
                clearance_m=float(report.distance),
                pair=report.pair,
                q_rad_or_m=[float(v) for v in q],
            )
            if worst is None or report.distance < worst["clearance_m"]:
                worst = record
            if len(examples) < 12:
                examples.append(record)
    ranges = {
        name: [float(model.lower[model.indices[b]]), float(model.upper[model.indices[b]])]
        for b, name in enumerate(model.names)
        if model.indices[b] >= 0
        and (
            name.startswith(("left_", "right_"))
            or name in ("torso_yaw", "torso_lower", "torso_upper", "torso")
        )
    }
    return dict(
        robot_revision=ROBOT_REVISION,
        status="failed" if worst else "sampled_clear",
        certification="none; finite axis sweeps, coupled wrist corners, seeded coupled poses",
        margin_m=world.margin,
        axis_samples=samples,
        seed=20261110,
        nominal_ranges=ranges,
        counts=dict(counts),
        failure_pairs=dict(pairs.most_common()),
        worst=worst,
        counterexamples=examples,
        unresolved=[
            "continuous coupled-space coverage",
            "vendor drive pigtail locations and intermediate shoulder branch routing",
            "cable fatigue, electrical qualification and mounting strength",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=29)
    parser.add_argument("--random-samples", type=int, default=128)
    args = parser.parse_args()
    report = audit(args.samples, args.random_samples)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(status=report["status"], counts=report["counts"], output=str(args.output)))
    )
    raise SystemExit(2 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
