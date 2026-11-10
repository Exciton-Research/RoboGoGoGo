"""Sample the practical forward manipulation range; not a safety certificate.

Run with python -m robogogogo.robot.validate_working_range --output report.json.
Hard joint limits and runtime collision guards are unchanged.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..urdf import loads
from .definition import ROBOT_REVISION
from .harness import harness_clearance
from .mobile_design import mobile_dual_arm_urdf, mobile_home
from .self_collision import mobile_collision_world
from .shoulder_wiring import branch_specs, cover_mass_specs, fixed_wiring_report

# Roll is outward on each side; all values radians.
WORKING_RANGES = {
    "waist": (-0.35, 0.35),
    "upper_roll": (0.0, 0.4),
    "shoulder": (0.25, 1.2),
    "elbow": (-1.6, -0.3),
    "wrist_roll": (-0.25, 0.25),
    "wrist_pitch": (-0.25, 0.25),
    "tool_roll": (-0.25, 0.25),
}


def configurations(model, count=512, seed=43):
    rng = np.random.default_rng(seed)
    for n in range(count):
        q = mobile_home(model, ("standing", "low", "reach")[n % 3])
        for side, sign in (("left", 1), ("right", -1)):
            for suffix, (lo, hi) in WORKING_RANGES.items():
                q[model.indices[model.names.index(f"{side}_{suffix}")]] = rng.uniform(lo, hi) * (
                    sign if suffix == "upper_roll" else 1
                )
        yield f"random:{n}", q
    # Coordinated base-height transitions with both arms moving. These are
    # discrete samples of trajectories, not a continuous swept-volume proof.
    for posture in ("low", "reach"):
        start = mobile_home(model, "standing")
        end = mobile_home(model, posture)
        for side, sign in (("left", 1), ("right", -1)):
            for suffix, value in (("upper_roll", sign * 0.25), ("shoulder", 1.0), ("elbow", -1.2)):
                end[model.indices[model.names.index(f"{side}_{suffix}")]] = value
        for t in np.linspace(0, 1, 41):
            yield f"transition:{posture}:{t:.3f}", start * (1 - t) + end * t


def audit(count=512, seed=43):
    model = loads(mobile_dual_arm_urdf()).model
    world = mobile_collision_world(model)
    counts = Counter()
    failures = []
    minimum = float("inf")
    for label, q in configurations(model, count, seed):
        rigid = world.clearance(q)
        cable = harness_clearance(model, q, world._poses(q), world)
        minimum = min(minimum, rigid.distance, cable.distance)
        counts["samples"] += 1
        counts["rigid_clear"] += int(rigid.distance >= world.margin)
        counts["harness_clear"] += int(cable.distance >= world.margin)
        if min(rigid.distance, cable.distance) >= world.margin:
            counts["both_clear"] += 1
        else:
            report = min((rigid, cable), key=lambda r: r.distance)
            failures.append(
                dict(case=label, pair=report.pair, clearance_m=report.distance, q=q.tolist())
            )
    fixed = fixed_wiring_report(model)
    return dict(
        revision=ROBOT_REVISION,
        status="sampled_clear" if not failures and fixed["valid"] else "failed",
        fixed_wiring=fixed,
        shoulder_branches=branch_specs(model),
        service_cover_mass_upper_bound_kg=sum(p["mass_kg"] for p in cover_mass_specs(model)),
        seed=seed,
        ranges_rad=WORKING_RANGES,
        roll_convention="positive outward; negate right upper_roll",
        counts=dict(counts),
        margin_m=world.margin,
        minimum_guard_equivalent_clearance_m=minimum,
        failures=failures,
        limitations=[
            "Finite samples, not continuous-space certification or full nominal joint range",
            "Cable routes are quasistatic geometric curves; no cable forces or fatigue model",
            "Shoulder branch geometry is modeled; vendor socket clocking and microloom flex ratings remain unverified",
            "Cover models are service envelopes with mating openings, not manufacturing CAD",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    report = audit(args.count, args.seed)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "counts")}))
    raise SystemExit(0 if report["status"] == "sampled_clear" else 2)


if __name__ == "__main__":
    main()
