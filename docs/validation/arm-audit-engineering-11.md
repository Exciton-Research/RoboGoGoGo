# Arm assembly audit — engineering 11

The selected embodiment retains seven actuated arm joints per side: three shoulder,
one elbow and three wrist axes. No joint function or catalog motor was removed.
The static presentation fingers still do not establish a complete gripper actuator
or grasp-contact implementation.

## Corrected findings

- Studio rendered an illustrative elbow reducer in addition to the catalog motor,
  plus decorative wrist caps. These duplicate motor-like assemblies were removed.
- Generic elbow and wrist shells overlapped the actuator region. Arm shells now
  use shared backend spans: upper arm 202–255 mm and forearm 75–190 mm. Studio and
  capsule guards read the same spans. Generic solid wrist cuffs were removed;
  actual fitted motor sleeves remain available in Exterior view.
- The old elbow-cap collision volume represented the removed decorative drum.
  It was removed; catalog drive, mounting parts and output shafts remain guarded.
- Elbow/wrist shafts were attached to the fixed parent. Each output is now a
  backend part on its rotating child, with fixture mass/inertia and a collision
  capsule. The final roll output reaches the gripper flange.
- The wrist-roll carrier attaches to the next stator's existing mounting rail,
  rather than crossing its rotating shaft. Front stator ties move behind the
  output face, preserving the 12 mm guard around rotating output links.
- Generic wrist backbone rods that passed through actuator housings were removed.
  Explicit joint-output connections replace them; the forearm rails start beyond
  the elbow module. The accepted shoulder serial arrangement remains unchanged.

Only each named output/own-actuator bearing interface is exempted. Opposite-arm
and remote-body checks remain active. Fixed parts sharing a link still require
manufacturing fit analysis; these capsule checks are not a supplier CAD audit.

## Validation and limits

Fourteen focused assembly/range/cover regressions pass, including seven actuators
per side, rotating output ownership, shared shell dimensions and remote guards.
Six pose-preview and actuator-driven whole-body reaching tests also pass (20
focused tests total). Studio production build and changed-file Python lint pass.
Exterior and Engineering views were checked in the restarted Studio preview.

Seed 42: all 106 practical poses and torso-transition samples clear. Seed 43:
532/594 complete poses clear; remaining failures involve upper-arm/chest,
hand/base, wrist/torso or cross-arm interference. Full results are adjacent in
`working-range-engineering-11.json`. The audit does not certify every hard-limit
pose or arbitrary continuous motion.

This remains a mechanical simulation concept. Bearing selection, shaft sizing,
mounting holes, tolerances, load/thermal ratings and an actuated gripper remain
unqualified. Seven arm joints are retained; wrist packaging is not yet an
integrated production wrist module.
