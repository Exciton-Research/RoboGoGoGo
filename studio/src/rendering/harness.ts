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
      for (const index of [0, points.length - 1]) {
        const adjacent = index === 0 ? 1 : index - 1;
        const tangent = points[adjacent].clone().sub(points[index]).normalize();
        const clamp = new THREE.Mesh(
          new THREE.CylinderGeometry(
            route.radius * 1.8,
            route.radius * 1.8,
            0.018,
            16,
          ),
          new THREE.MeshStandardMaterial({
            color: "#9aa6ad",
            metalness: 0.7,
            roughness: 0.4,
          }),
        );
        clamp.position.copy(points[index]);
        clamp.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          tangent,
        );
        group.add(clamp);
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
