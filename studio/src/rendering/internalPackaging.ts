import * as THREE from "three";
import type { RobotModel } from "../types";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Shared backend dimensions; inspection geometry is excluded from sensor views. */
export function addInternalPackaging(model: RobotModel, add: AddMesh) {
  const segment = (
    body: number,
    a: THREE.Vector3,
    b: THREE.Vector3,
    radius: number,
    color: string,
  ) => {
    const delta = b.clone().sub(a);
    const mesh = add(
      body,
      new THREE.CylinderGeometry(radius, radius, delta.length(), 48),
      color,
      a.clone().add(b).multiplyScalar(0.5).toArray(),
    );
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      delta.normalize(),
    );
    mesh.userData.inspectionOnly = true;
    mesh.visible = false;
    mesh.material.metalness = 0.6;
    return mesh;
  };
  for (const p of model.shoulder_packages) {
    const a = new THREE.Vector3(...p.start),
      b = new THREE.Vector3(...p.end);
    // Complete catalog motor/gearbox/driver module; do not invent vendor internals.
    segment(p.body, a, b, p.radius, "#34414a");
    const direction = b.clone().sub(a).normalize();
    for (const end of [a, b])
      segment(
        p.body,
        end.clone().addScaledVector(direction, -0.002),
        end.clone().addScaledVector(direction, 0.002),
        p.radius,
        "#a1aeb5",
      );
    // Custom coaxial output coupling between module and the joint carrier.
    const distance = p.name.includes("elbow") ? 0.025 : 0.075;
    segment(
      p.body,
      a.clone().addScaledVector(direction, -distance + 0.012),
      a,
      0.012,
      "#778d99",
    );
  }
  for (const p of model.cable_passages) {
    const a = new THREE.Vector3(...p.start),
      b = new THREE.Vector3(...p.end);
    const sleeve = segment(p.body, a, b, p.radius, "#46849b");
    sleeve.material.transparent = true;
    sleeve.material.opacity = 0.25;
    sleeve.material.depthWrite = false;
    segment(p.body, a, b, p.bundle_radius!, "#e6ae51");
    // Visible end ferrules mark the routing boundary, not an invented joint cable.
    const direction = b.clone().sub(a).normalize();
    for (const end of [a, b])
      segment(
        p.body,
        end.clone().addScaledVector(direction, -0.003),
        end.clone().addScaledVector(direction, 0.003),
        p.radius,
        "#263334",
      );
  }
}
