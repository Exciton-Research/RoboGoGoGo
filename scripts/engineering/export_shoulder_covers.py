"""Export split hollow shoulder cover concepts from simulation envelopes.

Usage: export_shoulder_covers.py envelope.json output_directory
Run with cadquery-ocp installed. Input uses metres, CAD output uses millimetres.
These are service-shell concepts, not manufacturing-released enclosures: no
fastener bosses, sealing features, tolerance stack or removal-path certification.
"""

import argparse
import json
from pathlib import Path


def export(specification, output):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
    )
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    def boolean(operation, a, b):
        result = operation(a, b)
        result.Build()
        if not result.IsDone() or not BRepCheck_Analyzer(result.Shape()).IsValid():
            raise ValueError("Invalid cover boolean result")
        return result.Shape()

    def capsule(item, key):
        a, b = (gp_Pnt(*(1000 * x for x in item[k])) for k in ("start", "end"))
        direction = gp_Dir(gp_Vec(a, b))
        radius = item[key] * 1000
        cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(a, direction), radius, a.Distance(b)).Shape()
        for point in (a, b):
            cylinder = boolean(
                BRepAlgoAPI_Fuse, cylinder, BRepPrimAPI_MakeSphere(point, radius).Shape()
            )
        return cylinder

    def union(shapes):
        result = shapes[0]
        for shape in shapes[1:]:
            result = boolean(BRepAlgoAPI_Fuse, result, shape)
        return result

    def volume(shape):
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, properties)
        return properties.Mass()

    output.mkdir(parents=True, exist_ok=True)
    reports = []
    bodies = sorted({cover["body"] for cover in specification["covers"]})
    for body in bodies:
        covers = [c for c in specification["covers"] if c["body"] == body]
        outer = union([capsule(c, "radius") for c in covers])
        cavity = union([capsule(c, "inner_radius") for c in covers])
        shell = boolean(BRepAlgoAPI_Cut, outer, cavity)
        # Split through the cavity centroid. Both halves are exported separately;
        # a 0.4 mm assembly seam prevents coincident faces and leaves an explicit gap.
        split_y = sum((c["start"][1] + c["end"][1]) * 500 for c in covers) / len(covers)
        halves = []
        for side, low, high in (("minus", -2000, split_y - 0.2), ("plus", split_y + 0.2, 2000)):
            box = BRepPrimAPI_MakeBox(gp_Pnt(-2000, low, -2000), gp_Pnt(2000, high, 2000)).Shape()
            half = boolean(BRepAlgoAPI_Common, shell, box)
            material = volume(half)
            if material <= 0:
                raise ValueError("Split plane failed to produce two shells")
            interference = volume(boolean(BRepAlgoAPI_Common, half, cavity))
            if abs(interference) > 1e-3:
                raise ValueError("Cover intrudes into its required cable cavity")
            name = specification["names"][body] + "_service_" + side
            writer = STEPControl_Writer()
            writer.Transfer(half, STEPControl_AsIs)
            if writer.Write(str(output / (name + ".step"))) != IFSelect_RetDone:
                raise ValueError("STEP export failed")
            halves.append(
                dict(name=name, volume_mm3=material, cavity_interference_mm3=interference)
            )
        reports.append(
            dict(
                body=specification["names"][body],
                halves=halves,
                seam_gap_mm=0.4,
                nominal_wall_mm=2.0,
                overlapping_envelopes_removed_mm3=sum(
                    volume(capsule(c, "radius")) - volume(capsule(c, "inner_radius"))
                    for c in covers
                )
                - volume(shell),
            )
        )
    report = dict(
        status="concept_geometry_valid",
        manufacturing_ready=False,
        scope="Valid B-rep shells, unioned cavities, separated halves; not an assembly or motion certificate",
        covers=reports,
        blockers=[
            "Mounting bosses and fasteners not designed",
            "Removal sweeps not validated",
            "Vendor connector positions and cable selection unresolved",
            "Inter-link mating openings and hardware cutouts not designed",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(json.loads(args.specification.read_text()), args.output)
