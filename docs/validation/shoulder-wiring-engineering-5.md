# Shoulder interconnects — engineering 5

The yaw, roll and pitch drives now have explicit adapter trays, fixed device leads, and two moving interconnects on each shoulder. Power feed-through terminals are separate from the local drive feed. The CAN chain and the power circuits reference the physical routes exported in `examples/robots/mobile_dual_arm.wiring.json` and Studio's `/api/snapshot` model data.

The yaw-to-roll microloom is 295 mm long and roll-to-pitch is 250 mm. Both use a 16 mm bundle envelope, a 40 mm minimum bend requirement, and a conservative twist limit of pi radians/m. These are **custom microloom requirements, not a verified supplier cable rating**. Each fixed adapter lead has explicit 10 mm fillets and a cut length of roughly 100–140 mm. Ten fixed leads are checked against their own host's motor, connectors, mounting rods, neighbouring wires and cover cavities. Same-host interference is not silently exempted.

The two AK80-64 motors use separate XT30 power and GH125 CAN interfaces, while the AK60-39 V3 yaw drive uses a combined XT30 2+2 interface. This distinction follows the [manufacturer cable compatibility table](https://www.cubemars.com/product/cables.html). The custom junction blocks accommodate separate downstream feeds; they are not misrepresented as the manufacturer's four-pin device connector. **Vendor socket coordinates, clocking, and contact numbering remain unverified.** The modeled rear-face placement is an adapter design assumption, not a measured vendor CAD result.

## Geometry and motion checks

The covered service channels have fixed dimensions and link attachments. They do not expand to follow the cable pose. The guard checks internal cables against solid motor cylinders and installation geometry, then verifies containment within the inner cover cavities. Internal installation clearance is 1 mm; rigid robot and external harness guards retain their 12 mm margin. The reported combined clearance is guard-equivalent, not one uniform physical gap. Named termination boots have a bounded 28 mm connection zone; the remaining cable still checks its own junction.

The same geometry drives Studio's cable rendering and motion guard. Draft pose previews also check the wiring before enabling Apply; a rigidly clear pose cannot hide a cable or cover-fit failure. Covered view shows the service shells; Engineering view removes them to expose the adapters and interconnects. All three shoulder motors remain present. Fixed fittings contribute mass/inertia at their actual host frames. Two-millimetre ABS shell envelopes use 1200 kg/m³ and conservatively count overlapping seams, adding an upper-bound total of about 3.02 kg across both shoulders. This is not a production mass estimate.

- Seed 43: 594 poses and transition samples, 594 harness passes and 593 overall passes.
- Independent seed 44: 338 poses and transition samples, 338 harness passes and 337 overall passes.
- Both retained failures are the left hand hitting the wheeled base in a low posture. The guard correctly rejects them.
- Fixed-lead installation checks pass, with a minimum modeled gap above 1 mm.

Reproduce either audit:

```sh
.venv/bin/python -m robogogogo.robot.validate_working_range --output /tmp/shoulder.json
.venv/bin/python -m robogogogo.robot.validate_working_range --seed 44 --count 256 --output /tmp/shoulder-holdout.json
```

The audits intentionally return exit code 2 because each includes a legitimate base collision. Full configurations are retained in the JSON reports next to this file. Joint limits were not narrowed to obtain these results.

## Release boundary

This completes the modeled interconnections between the three shoulder adapters. It does not release a manufacturing loom: vendor connector placement, supplier bending/fatigue qualification, protection ratings, assembly access, shell mating details and strain-relief hardware still need physical validation. The covers are service envelopes with open mating interfaces, not detailed manufacturing surfaces. Incoming torso distribution and downstream arm wiring outside these shoulder interconnects remain separate work. No real robot was commanded.
