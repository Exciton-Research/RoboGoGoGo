# Compact shoulder — engineering 7

The reference image guides the arrangement and appearance, not the internal
transmission or supplier dimensions. Each shoulder retains three catalog motors:
AK60-39 yaw, AK80-64 lift, AK80-64 upper-arm roll. The serial axes are now Z-Y-X.
Legacy IDs `waist`, `upper_roll`, and `shoulder` remain, but mean yaw, lift, and
axial roll respectively. Studio displays those functional labels. **Engineering-6
joint trajectories must not be interpreted as engineering-7 trajectories.**

The yaw motor offset is 65 mm, lift offset 95 mm, and axial roll offset 90 mm.
Shoulder roots moved from +/-400 mm to +/-350 mm. The upper-arm sleeve radius is
55 mm; its visible distal cover starts after the axial motor. The roll motor case
follows the lift carrier, while the output collar follows the rotating arm.

Each shoulder stator has a custom fork carrier: two 8 x 10 mm cheeks and a bridge.
The axial motor's bridge is proximal, leaving its distal output open. These solid
sections, catalog motor bounds, and 2 mm split covers drive the rendered model and
mass calculation. Cheek meshes fit their conservative collision capsules. Their
nominal radial gap to the fitted cover exceeds 4 mm. White end bands and dark blue
housings follow the supplied visual reference; they are custom cosmetic finishes.
Engineering view removes covers to reveal the motor and carrier assembly.

The axial motor sits inside its own upper-arm assembly, so its named local
containment pair is exempted. Yaw and lift drives still check the moving upper
arm, other shoulder parts, and the opposite arm. No global arm/torso exemption or
joint-limit reduction was added. Existing torso support and runtime trajectory
checks remain active. Detailed electrical packaging remains out of scope.

## Validation

- Seed 43: 594 samples; all 594 shoulder pair checks pass, 589 complete robot poses pass.
- Five low poses put a hand near or through the wheeled base; the guard rejects them.
- Independent seed 44: all 338 samples pass both shoulder and complete robot checks.
- Smallest sampled shoulder pair gap: 12.738 mm, above the unchanged 12 mm reserve.
- Eight packaging/cover/range regressions pass, including rejection of hand/base
  and hand/hand interference and preservation of the elbow pivot during axial roll.
- Six pose-preview/whole-body tests pass, including coordinated wheel, torso and
  two-arm actuator-driven reaching. Studio production build passes.

Reproduce the audits with `python -m robogogogo.robot.validate_working_range` and
`--seed 44 --count 256`, specifying `--output` for each report. Complete results
are in the adjacent engineering-7 JSON files. Shoulder pair counts cover the
retained mechanical pair checks, not an independent detailed CAD contact solver.

This is a checked simulation concept, not a fabrication release. Supplier-specific
mounting holes, bearing fits, actual output coupling geometry, fasteners, structural
strength, tolerances and thermal behavior remain unqualified. The finite audits do
not certify the complete nominal joint range or all possible trajectories.
