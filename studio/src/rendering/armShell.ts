import * as THREE from "three";

/** Smooth split shell within the existing arm envelope, in local Y coordinates.
 * Profile interpolation is bounded: mesh refinement cannot enlarge the radius.
 * The inner return closes each profile; narrow axial seams expose the dark carrier.
 */
export function armShell(length: number, radius: number, half: number) {
  const knots = [
    [0.018, 0.69],
    [0.052, 0.95],
    [length * 0.38, 0.96],
    [length - 0.065, 0.79],
    [length - 0.025, 0.6],
  ];
  const profile: THREE.Vector2[] = [];
  for (let segment = 0; segment < knots.length - 1; segment++) {
    const [x0, r0] = knots[segment];
    const [x1, r1] = knots[segment + 1];
    for (let i = 0; i < 12; i++) {
      const t = i / 12;
      const blend = t * t * (3 - 2 * t);
      profile.push(
        new THREE.Vector2(
          radius * (r0 + (r1 - r0) * blend),
          x0 + (x1 - x0) * t,
        ),
      );
    }
  }
  const [end, endRadius] = knots[knots.length - 1];
  profile.push(new THREE.Vector2(radius * endRadius, end));
  const inner = profile
    .map((p) => new THREE.Vector2(p.x - 0.003, p.y))
    .reverse();
  profile.push(...inner, profile[0].clone());
  return new THREE.LatheGeometry(
    profile,
    48,
    half * Math.PI + 0.009,
    Math.PI - 0.018,
  );
}
