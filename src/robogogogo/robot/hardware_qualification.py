"""Vendor evidence and engineering release checks, separate from simulation guards.

The Igus parts are comparison candidates, not a released internal-loom BOM.
A geometric pass under custom assumptions must not certify real hardware.
"""

import math

CABLE_CATALOG = "https://toolbox.igus.com/wp-content/uploads/2023/12/robotics-catalog-2023.pdf"
CABLE_CANDIDATES = (
    dict(part="CFROBOT2.07.04.C", diameter_m=0.008, printed_pages=[176, 177]),
    dict(part="CFROBOT8.022", diameter_m=0.0075, printed_pages=[194, 196]),
)


def candidate_bend_limit(bundle_radius):
    """Conservative bundle-centre radius with each cable on the inside of a bend.

    Published twisted minimum is 10*d. Add the bundle radius and cable radius
    rather than assuming the supplier's radius convention grants extra space.
    This is a fit screening calculation, not a cable-life qualification.
    """
    if not math.isfinite(bundle_radius) or bundle_radius < 0:
        raise ValueError("Bundle radius must be finite and non-negative")
    return max(10.5 * c["diameter_m"] + bundle_radius for c in CABLE_CANDIDATES)


def shoulder_qualification(model):
    from .shoulder_wiring import branch_specs

    branches = []
    for branch in branch_specs(model):
        required = candidate_bend_limit(branch["radius"])
        branches.append(
            dict(
                name=branch["name"],
                modeled_minimum_bend_m=branch["bend_limit_m"],
                candidate_required_bend_m=required,
                specification_compatible=branch["bend_limit_m"] >= required,
            )
        )
    return dict(
        status="not_qualified",
        hardware_ready=False,
        source=CABLE_CATALOG,
        cable_candidates=list(CABLE_CANDIDATES),
        branches=branches,
        blockers=[
            "Internal loom has no released supplier part or flex-life rating; candidate bend requirement is 92 mm, modeled requirement is 40 mm.",
            "AK60-39 CAD dimensions verified; connector-to-robot mounting transform remains unverified.",
            "AK80-64 STEP is a single simplified solid without connector geometry; rear socket locations remain unverified.",
            "Fixed leads use an unqualified 10 mm bend requirement; vendor accessory specifications give no bend-life rating.",
            "Cover assembly, fastener access, thermal performance and manufacturing tolerances require validation.",
        ],
        scope="Engineering evidence only; does not enable or disable simulation or real-robot commands",
    )
