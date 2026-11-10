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
    const allowance = p.cover_allowance_m ?? 0.004;
    const skin = add(
      p.body,
      new THREE.CylinderGeometry(
        p.radius + allowance,
        p.radius + allowance,
        b.distanceTo(a) + 2 * allowance,
        64,
      ),
      "#e4e6e2",
      a.clone().add(b).multiplyScalar(0.5).toArray(),
    );
    skin.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    skin.userData.removableCover = true;
    skin.material.roughness = 0.38;
    if (p.shoulder_axis) {
      // A segmented shoulder housing encloses each real motor. These necks
      // remain within the existing socket collision envelope and follow each
      // motor's actual parent; a single rigid shell would bind the three axes.
      const neck = segment(
        p.body,
        a
          .clone()
          .addScaledVector(direction, -(p.mounting_offset ?? 0.025) + 0.035),
        a,
        0.026,
        "#e4e6e2",
      );
      neck.userData.inspectionOnly = false;
      neck.userData.removableCover = true;
      neck.visible = true;
      neck.material.metalness = 0.15;
    }
    // Custom coaxial output coupling between module and the joint carrier.
    const distance = p.mounting_offset ?? 0.025;
    segment(
      p.body,
      a.clone().addScaledVector(direction, -distance + 0.012),
      a,
      0.012,
      "#778d99",
    );
  }
  for (const p of model.assembly_parts ?? []) {
    const fitting = segment(
      p.body,
      new THREE.Vector3(...p.start),
      new THREE.Vector3(...p.end),
      p.radius,
      p.kind === "connector" || p.kind === "wire" ? "#232b30" : "#9aa8b1",
    );
    if (p.kind === "connector") {
      fitting.geometry.dispose();
      fitting.geometry = new THREE.BoxGeometry(
        p.box_size?.[0] ?? Math.min(0.014, p.radius * 1.5),
        new THREE.Vector3(...p.end).distanceTo(new THREE.Vector3(...p.start)),
        p.box_size?.[2] ?? Math.min(0.01, p.radius),
      );
      fitting.material.metalness = 0.05;
    }
    if (p.kind === "clamp" || p.kind === "swivel_clamp") {
      fitting.geometry.dispose();
      const ring = new THREE.TorusGeometry(
        p.kind === "swivel_clamp" ? 0.02 : 0.0185,
        p.kind === "swivel_clamp" ? 0.002 : 0.0035,
        12,
        40,
      );
      ring.rotateX(Math.PI / 2);
      fitting.geometry = ring;
      fitting.material.color.set("#7d919c");
    }
    // Saddles and plugs sit outside the module cover and remain physical in RGB.
    fitting.userData.inspectionOnly = false;
    fitting.visible = true;
  }
  for (const cover of model.shoulder_covers ?? []) {
    const a = new THREE.Vector3(...cover.start),
      b = new THREE.Vector3(...cover.end);
    const skin = add(
      cover.body,
      new THREE.CapsuleGeometry(cover.radius, a.distanceTo(b), 12, 40),
      "#e4e6e2",
      a.clone().add(b).multiplyScalar(0.5).toArray(),
    );
    skin.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      b.clone().sub(a).normalize(),
    );
    skin.userData.removableCover = true;
    skin.material.roughness = 0.4;
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
