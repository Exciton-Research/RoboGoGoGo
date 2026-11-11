# Shoulder interconnects — engineering 5

**Historical design, superseded in engineering-6.** At the user’s request, the active
robot now omits detailed cables, connectors, electrical supports and their guards.
The prior electrical studies below remain reference material, not the Studio model.
Motor saddles, catalog actuator envelopes, mechanical collision checks and support
monitoring remain active. Studio uses slim split sleeves with 1 mm nominal motor
clearance, 2 mm walls and open ends; Engineering hides these sleeves. Their total
modeled mass is 0.558 kg across all 14 arm actuators. Production fasteners, tolerances
and electrical routing remain deferred. The engineering-6 working-range report
records 593 clear poses out of 594; the remaining pose has hand/base interference.
Coordinated wheel/torso/two-arm motion, draft collision rejection and both visual
modes were verified after this simplification. The old STEP cover concepts below
are not the current slim sleeves.

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

## Vendor inspection and cover CAD follow-up (2026-11-10)

Downloaded sources and SHA-256 fingerprints are recorded in `vendor/sources.json`.
The reproducible `scripts/engineering/inspect_step.py` reader measured the official
AK60-39 V3 assembly at 79 x 79 x 67 mm and AK80-64 at 61.9 x 98 x 98 mm in their
respective native STEP axes. These confirm the catalog envelopes, not connector
mounting transforms. AK60 contains connector-named components, but their location
has **not** been registered to our robot frame. AK80 is one simplified solid;
its socket positions cannot be recovered from that download. Its internal STEP
product label also differs from the advertised filename, so identity should be
confirmed with the supplier before tooling.

The [AK54-4810-2D-A2 manual](https://www.cubemars.com/data/cms/202610/ak54-4810-2d-a2-driver-installation-instructions.pdf),
pages 6-9, identifies the yaw driver's AMASS XT30PW(2+2)-M interface and mating
XT30(2+2)-F. The supplied lead is specified as 300 +/- 10 mm. This does not validate
our custom shorter leads, their socket clocking, or a 10 mm bend radius.

The [Igus robotics catalog](https://toolbox.igus.com/wp-content/uploads/2023/12/robotics-catalog-2023.pdf),
printed pages 176-177 and 194-196, lists 8 mm CFROBOT2.07.04.C and 7.5 mm
CFROBOT8.022, both requiring 10*d for twisted flexing. A conservative internal
bundle screening radius is 92 mm (80 mm plus 8 mm bundle offset plus 4 mm cable
radius). The modeled 40 mm requirement is incompatible with those candidates.
They are **not** selected as a released internal loom. Existing external routing
retains its larger, conservative 8.5 mm power-cable allowance. Studio's Engineering
note and model API now report hardware qualification separately from simulation
geometry validity; these checks do not enable hardware execution.

`examples/robots/cad/shoulder-cover-concepts.zip` contains 12 hollow STEP halves.
The exporter unions same-host cover envelopes and cavities before cutting the
shell, eliminating redundant internal walls. A 0.4 mm split gap separates the
halves; nominal walls are 2 mm. All halves passed B-rep validity, zero cavity-volume
interference, and STEP reimport volume checks (0.01 mm3 tolerance). The report is
`shoulder-cover-cad.json`. Shell-only mass at the existing density assumption is
2.017 kg. This excludes bosses, screws, openings and production tolerances.

These are **offline cover concepts**, not replacements for the conservative
runtime envelopes or their inertia. They lack motor/connector openings, mounting
bosses, fasteners and validated removal sweeps. A valid hollow solid is not proof
of assemblability or collision-free robot movement. No vendor socket position or
full working-range hardware clearance is newly certified by this work.

Reproduce using an isolated environment with `cadquery-ocp==8.0.1.1.0`:

```sh
python scripts/engineering/inspect_step.py /path/to/vendor.step --output /tmp/vendor-measurement.json
python scripts/engineering/export_shoulder_covers.py examples/robots/cad/shoulder-cover-envelopes.json /tmp/shoulder-covers
```

The fixed-lead generator also now validates both connector exit legs, rejects
invalid directions and impossible reversals, and handles straight routes without
NaNs. Validation: 12 focused wiring/qualification tests passed; Studio production
build passed. Existing motion-audit counts above remain historical engineering-5
results, not a CAD-based qualification of the new cover concepts.
