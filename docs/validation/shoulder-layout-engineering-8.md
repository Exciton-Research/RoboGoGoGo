# Shoulder layout correction — engineering 8

The supplied close-up replaces the earlier interpretation: a torso-side actuator,
a cross-axis shoulder actuator in a fork, and an axial upper-arm actuator. The
photo guides the arrangement; hidden vendor mechanisms are not established by it.

The actual kinematic chain is now Y-Z-X. Legacy `waist` is shoulder lift,
`upper_roll` is shoulder swing, and `shoulder` is axial roll. The lift AK80-64
mounts 120 mm inboard of the shoulder pivot, toward the torso. The swing AK60-39
mounts 70 mm along the cross axis and follows lift. The axial AK80-64 retains its
90 mm offset. Catalog dimensions remain unchanged; mass, joint effort, fixture
inertia, exported URDF and Studio controls follow the revised placement. Fork
cheeks flank the swing axis instead of crossing the upper arm. Engineering view
reveals the torso-side motor. The upright external motor is no longer present.

Engineering-7 recordings are not compatible with these changed joint semantics.
This revision is a simulation layout, not a manufacturing release: shaft support,
fasteners, bearings, actual coupling geometry and structural qualification remain
unfinished. Detailed cables remain deferred at the user's request.

## Verification

- Nine packaging, cover and practical-motion tests pass, including exactly three
  motors, inboard lift placement, clear sampled height transitions and rejection
  of a deliberately inward-folded arm that intersects the chest.
- Six pose-preview and coordinated wheel/torso/two-arm control tests pass.
- Studio production build and changed-file Python lint pass; both display modes
  were inspected in the running preview.
- Seed 42: all 106 practical pose/transition samples clear.
- Broader seed 43 audit: 516 of 594 complete poses clear. The 78 failing samples
  involve arm/chest, hand/base or cross-arm interference and remain guarded.
  521 samples clear all shoulder-related pairs, including arm/chest pairs.
  No failure's minimum pair was motor/motor. This does not certify every motor
  pair independently or prove a continuous collision-free workspace.

The broader audit is retained in `working-range-engineering-8.json`, including
failed configurations. It must not be described as a completely clear workspace.
Reproduce with `python -m robogogogo.robot.validate_working_range --output report.json`.
