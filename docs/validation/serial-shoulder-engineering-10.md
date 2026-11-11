# Corner-centered serial shoulder — engineering 10

The cross-axis swing motor is centered on the support/upper-arm junction, with
its Z axis unchanged. The earlier -60 mm rear offset is removed. The axial roll
actuator moves 20 mm distally (110 mm mounting offset), leaving clearance for the
centered motor. The upper-arm carrier begins at that actuator, and the distal
cover starts at 202 mm. Its conservative collision capsule covers the complete
202–320 mm sleeve; proximal actuator geometry is checked separately.

The structural sequence now follows the articulated chain:

1. Torso backbone rails connect to the inboard stator of the lift motor.
2. The lift output connects to the swing motor's fork cheek.
3. The swing output connects through a central carrier to the roll motor's fork.
4. The roll output connects to the upper arm.

The torso attachment starts at y=+/-25 mm, z=200 mm in the torso frame and ends
at the lift motor's inboard face. Its inner end is lower to preserve head clearance.
The old diagonal torso-to-shoulder bypass rails and decorative shoulder shafts
are removed. All new connections are shared backend parts, rendered in Studio,
with mass and inertia assigned to their actual parent or rotating output frame.
The swing fork opens on its output side. The roll fork cheeks lie above/below its
axis so the central carrier avoids sweeping into the inboard lift mount.

Collision exclusions name only each output shaft and its own actuator package
for the bearing interface. Opposite-arm and remote-body checks remain active.
Same-link fixtures are rigid assembly geometry, not independently checked moving
pairs; manufacturing fit, bearing support, fasteners and strength are unqualified.

## Validation

- The 106 seed-42 poses and torso transitions clear the complete robot guard.
- All 594 seed-43 poses clear the 542 retained shoulder motor/carrier pair checks.
  Their minimum clearance is 14.067 mm against the 12 mm guard.
- The broader whole-robot audit clears 522/594 poses. Other poses encounter arm/
  chest, hand/base or cross-arm interference; these remain guarded. This is not
  a collision-free certificate for every hard-limit pose or arbitrary motion.
- Regression checks cover the corner center, unchanged axes, serial parent/output
  ownership, output-bearing exclusions and retained opposite-arm checks.
- All 18 focused tests pass, including actuator-driven coordinated wheel, torso
  and two-arm reaching on the final geometry.
- Studio production build and changed-file lint pass. Engineering view was checked.

Engineering-9 recordings must not be replayed as engineering-10 joint trajectories.
