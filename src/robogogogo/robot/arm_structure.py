"""Arm shell spans between catalog actuators, shared with Studio and guards.

These are assembly envelopes, not supplier CAD or qualified structural sections.
"""

ARM_STRUCTURE = {
    "shoulder": dict(
        sleeve_start=0.202, sleeve_end=0.255, carrier_start=0.11, carrier_end=0.255, radius=0.055
    ),
    "elbow": dict(
        sleeve_start=0.075, sleeve_end=0.19, carrier_start=0.06, carrier_end=0.1985, radius=0.065
    ),
}


def arm_structure_specs(model):
    return [
        dict(body=model.names.index(f"{side}_{suffix}"), **spec)
        for side in ("left", "right")
        for suffix, spec in ARM_STRUCTURE.items()
    ]
