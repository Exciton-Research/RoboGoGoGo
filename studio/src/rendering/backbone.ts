import * as THREE from "three";
import type { RobotModel } from "../types";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Custom structural concept: open rails connect real joint origins, not cable ends. */
export function addBackbone(model: RobotModel, add: AddMesh) {
  const beam = (
    body: number,
    start: THREE.Vector3,
    end: THREE.Vector3,
    radius: number,
  ) => {
    const delta = end.clone().sub(start);
    const mesh = add(
      body,
      new THREE.CylinderGeometry(radius, radius, delta.length(), 24),
      "#667c89",
      start.clone().add(end).multiplyScalar(0.5).toArray(),
    );
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      delta.normalize(),
    );
    mesh.userData.inspectionOnly = true;
    mesh.visible = false;
    mesh.material.metalness = 0.7;
  };
  model.names.forEach((name, body) => {
    if (name.endsWith("actuator") || name === "world") return;
    // Shoulder support is the serial motor carrier from the backend, not a
    // diagonal torso rail terminating beyond the first actuator.
    if (/^(left|right)_pedestal$/.test(name)) return;
    const parent = model.parents[body],
      end = new THREE.Vector3(...model.joint_origins[body]);
    if (parent < 0 || end.length() < 0.1) return;
    if (
      /^(left|right)_(wrist_roll|wrist_pitch|tool_roll)$/.test(
        model.names[parent],
      )
    )
      return;
    for (const side of [-1, 1]) {
      const offset = new THREE.Vector3(0, side * 0.025, 0);
      const start = offset.clone();
      if (/^(left|right)_shoulder$/.test(model.names[parent])) start.x = 0.11;
      if (/^(left|right)_elbow$/.test(model.names[parent])) start.x = 0.075;
      beam(parent, start, end.clone().add(offset), 0.009);
    }
    beam(
      parent,
      end.clone().add(new THREE.Vector3(0, -0.025, 0)),
      end.clone().add(new THREE.Vector3(0, 0.025, 0)),
      0.012,
    );
  });
  const base = model.names.indexOf("base_link");
  for (const y of [-0.27, 0.27])
    beam(
      base,
      new THREE.Vector3(-0.3, y, 0),
      new THREE.Vector3(0.3, y, 0),
      0.024,
    );
  for (const x of [-0.25, 0.25])
    beam(
      base,
      new THREE.Vector3(x, -0.27, 0),
      new THREE.Vector3(x, 0.27, 0),
      0.024,
    );
}
