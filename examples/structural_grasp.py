"""Run: python examples/structural_grasp.py --material fragile_shell --recipe crush."""

import argparse
import json
from pathlib import Path

import jax

from robogogogo.simulation.structural_lab import StructuralLab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material", choices=["soft_foam", "soft_elastomer", "fragile_shell"], default="soft_foam"
    )
    parser.add_argument("--recipe", choices=["bend", "tear", "crush"], default="bend")
    parser.add_argument("--output", type=Path, default=Path("structure.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    lab = StructuralLab(args.material, args.recipe)
    while lab.time < 6:
        frame = lab.step()
    args.output.write_text(json.dumps(lab.export()))
    print(
        f"{len(frame['fragments'])} pieces; {frame['broken_bonds']} broken bonds; "
        f"{frame['peak_pressure_pa'] / 1000:.1f} kPa peak pressure; {args.output}"
    )


if __name__ == "__main__":
    main()
