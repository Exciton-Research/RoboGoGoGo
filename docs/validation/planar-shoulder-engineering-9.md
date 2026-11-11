# Planar shoulder motor placement — engineering 9

The upper-right actuator in the user's engineering-view reference is the
cross-axis swing motor. Its center moves from (0, 0, 103.5) mm to (-60, 0, 0) mm
in the lift frame. Its rotation axis remains Z. This places its center in the
lift-frame XY plane containing the other two motor centers at neutral swing.
Both shoulders use the correction. This is a mounting relationship at neutral
swing, not a promise that all centers remain in a fixed world plane during motion.

The swing pivot moves with the motor for direct actuation. A compensating +60 mm
origin on the axial-roll joint preserves the other motor centers and arm home
pose. Swing trajectories change because their pivot moved. URDF motor mass,
fixtures, covers, collision envelopes and Studio use the changed transforms.
Engineering-8 trajectories are incompatible with the new geometry.

Studio no longer draws the old long swing shaft/neck into the newly centered
motor. The axial output collar now subtracts the child origin before drawing in
the child frame, preventing a doubled 60 mm displacement.

The regression verifies all three motor centers in the common lift frame,
unchanged lift/axial motor centers, and the swing motor's new position. Nine
packaging/cover tests and the practical-range regression pass. The production
Studio build passes, and the new layout was inspected in Engineering view.
Six pose-preview and actuator-driven whole-body reach tests also pass.

Seed 42 samples: 96 of 106 complete poses clear. Ten inward poses violate the
12 mm guard through arm/chest or mounting-part interference and are explicitly
retained as rejection tests. The adjacent JSON preserves these configurations.
This relocation is not a collision-free full-range or manufacturing qualification.
Bearing support, output coupling details, fasteners and structural analysis still
need engineering validation.
