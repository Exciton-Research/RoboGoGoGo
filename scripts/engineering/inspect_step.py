"""Measure a locally downloaded vendor STEP without adding CAD to runtime dependencies.

Install cadquery-ocp in an isolated engineering environment. No network access or
source modification occurs here. Bounds stay in the STEP reader's millimetre
coordinate system; they are NOT a robot-frame connector placement.
"""

import argparse
import hashlib
import json
from pathlib import Path


def inspect(path):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError(f"Cannot read STEP: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP contains no transferred shape")
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds)
    low, high = list(bounds.CornerMin().Coord()), list(bounds.CornerMax().Coord())
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = 0
    while explorer.More():
        solids += 1
        explorer.Next()
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_bounds_mm=[low, high],
        extent_mm=[b - a for a, b in zip(low, high)],
        solids=solids,
        connector_placement_verified=False,
        scope="Native bounds only; no mating-face, clocking or assembly transform inference",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.step)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
