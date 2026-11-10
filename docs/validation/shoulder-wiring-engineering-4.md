# Engineering 4: shoulder clearance and arm wiring

The arm roots move 50 mm outward on each side. Yaw and pitch drives have separate axial offsets; the roll drive mounts aft of the shoulder. Collision checks now include the proximal drive housings against the moving upper arms. Catalog cylinders and their 4 mm cover allowance retain enclosing collision proxies and the existing 12 mm guard margin. Each shoulder contains exactly three catalog motors (yaw, roll, pitch) inside segmented covers; duplicate illustrative shoulder actuator details have been removed. The shoulder housing articulates with those axes instead of using a single rigid enclosure.

Rear shoulder harness supports run behind the roll drives. Shoulder and elbow bundles land on separate planes of the upper arm. Shoulder fairleads have a 45-degree angular envelope, shown in Studio using the backend guide directions. Cable length and bend limits remain enforced. Curve selection is quasistatic and discrete; it does not model cable loads, fatigue, or guarantee continuous motion between sampled configurations.

The unchanged development sample (seed 42, 180 independently coupled poses across standing, low and reach postures) improved from 61/180 clear to 180/180. The independent seed-43 audit has 512 coupled poses plus 82 coordinated arm/torso transition samples: 593/594 clear, with all 594 harness checks clear. The remaining configuration puts the left hand into the wheeled base in a low posture; its full configuration is retained in `working-range-engineering-4.json`. It must remain blocked. These results do not certify the full Cartesian product of joint limits.

Reproduce the independent audit:

```sh
.venv/bin/python -m robogogogo.robot.validate_working_range --output /tmp/working-range.json
```

Exit code 2 deliberately signals the retained base collision. The report contains the tested joint ranges and counterexample.

## Wiring status

`robot.wiring.robot_wiring(model)` supplies the logical topology in Studio's model response: fourteen arm drives, separate 48 V proximal and 24 V wrist feeds, independent left/right CAN buses, unique addresses and two 120-ohm terminations per bus. Tests inject opens, crossed CAN lines, supply shorts, missing devices, duplicate addresses and misplaced terminations. Fixed adapter pigtails, panel connectors, supports and moving grouped harness spans have modeled geometry.

This is not a completed production loom. Intermediate branches to individual shoulder drives are not fully routed in 3D. Adapter termination points are design assumptions, not verified vendor port locations. Connector cavity numbering, protection ratings, crimping, strain relief, assembly access, load-bearing mounts and cable endurance still require hardware qualification. Torso/head/base electrical integration is outside this arm netlist. The existing 1 kg-per-hand static load check remains distinct from the unmet 15% torque reserve target.
