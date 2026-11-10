import * as THREE from "three";
import type { HarnessRoute } from "../types";

/** Backend-generated centreline points; never interpolate between commanded poses. */
export function createHarnessVisuals(scene: THREE.Scene) {
  const group = new THREE.Group();
  scene.add(group);
  let last: HarnessRoute[] | undefined;
  const clear = () => {
    group.children.forEach((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
    });
    group.clear();
  };
  const update = (routes: HarnessRoute[] | undefined) => {
    if (routes === last) return;
    last = routes;
    clear();
    for (const route of routes ?? []) {
      const points = route.points.map((p) => new THREE.Vector3(...p));
      const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
      const cable = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 64, route.radius, 8, false),
        new THREE.MeshStandardMaterial({
          color: route.valid ? "#263b49" : "#c65643",
          roughness: 0.72,
        }),
      );
      cable.castShadow = true;
      group.add(cable);
      if (route.name.includes("_shoulder_")) {
        for (const [i, point] of [
          points[0],
          points[points.length - 1],
        ].entries()) {
          const normal = route.guide_normals?.[i];
          if (!normal) continue;
          const guide = new THREE.Mesh(
            new THREE.TorusGeometry(0.017, 0.002, 12, 40),
            new THREE.MeshStandardMaterial({
              color: "#a1b3bd",
              metalness: 0.7,
              roughness: 0.35,
            }),
          );
          guide.position.copy(point);
          guide.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(...normal).normalize(),
          );
          group.add(guide);
        }
      }
    }
  };
  return {
    update,
    dispose: () => {
      clear();
      scene.remove(group);
    },
  };
}
