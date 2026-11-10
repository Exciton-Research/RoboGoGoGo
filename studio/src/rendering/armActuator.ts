import * as THREE from "three";
import type { RobotModel } from "../types";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Concept output-side bearing/reducer assembly, not a manufacturing motor CAD.
 * Housing is fixed to the parent, flange to the child. No animation or commanded
 * pose is used: both inherit the simulated link transforms.
 */
export function addArmActuator(model: RobotModel, body: number, add: AddMesh) {
  const name = model.names[body];
  const elbow = name.endsWith("elbow");
  const radius = elbow ? 0.052 : 0.033;
  const offset = elbow ? 0.065 : 0.071;
  const side =
    name.startsWith("right_") && (elbow || name.endsWith("shoulder")) ? -1 : 1;
  const axis = new THREE.Vector3(...model.axes[body]).multiplyScalar(side);
  const align = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    axis,
  );
  const rows = model.joint_orientations[body];
  const rest = new THREE.Matrix4().set(
    ...([...rows[0], 0, ...rows[1], 0, ...rows[2], 0, 0, 0, 0, 1] as Parameters<
      THREE.Matrix4["set"]
    >),
  );
  const restRotation = new THREE.Quaternion().setFromRotationMatrix(rest);
  const origin = new THREE.Vector3(...model.joint_origins[body]);
  const part = (
    geometry: THREE.BufferGeometry,
    color: string,
    distance: number,
    fixed: boolean,
    radial = new THREE.Vector3(),
  ) => {
    const position = axis
      .clone()
      .multiplyScalar(distance)
      .add(radial.clone().applyQuaternion(align));
    const rotation = align.clone();
    if (fixed) {
      position.applyQuaternion(restRotation).add(origin);
      rotation.premultiply(restRotation);
    }
    const mesh = add(
      fixed ? model.parents[body] : body,
      geometry,
      color,
      position.toArray(),
    );
    mesh.quaternion.copy(rotation);
    mesh.material.metalness = color === "#a5afb3" ? 0.8 : 0.25;
    mesh.material.roughness = 0.36;
    return mesh;
  };
  // Stepped reducer casing, seal, output bearing flange and central shaft plug.
  part(
    new THREE.CylinderGeometry(radius, radius * 0.92, 0.016, 64),
    "#465158",
    offset,
    true,
  );
  part(
    new THREE.CylinderGeometry(radius * 0.98, radius * 0.98, 0.003, 64),
    "#131c22",
    offset + 0.0095,
    true,
  );
  part(
    new THREE.CylinderGeometry(radius * 0.88, radius * 0.88, 0.006, 64),
    "#a5afb3",
    offset + 0.014,
    false,
  );
  part(
    new THREE.CylinderGeometry(radius * 0.33, radius * 0.33, 0.002, 48),
    "#263334",
    offset + 0.018,
    false,
  );
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 4;
    const radial = new THREE.Vector3(
      Math.cos(angle) * radius * 0.66,
      0,
      Math.sin(angle) * radius * 0.66,
    );
    part(
      new THREE.CylinderGeometry(0.0025, 0.0025, 0.0015, 6),
      "#263334",
      offset + 0.0178,
      false,
      radial,
    );
  }
  // Separate index marks make housing/flange relative rotation visible.
  part(
    new THREE.BoxGeometry(0.003, 0.001, 0.008),
    "#db9157",
    offset + 0.0085,
    true,
    new THREE.Vector3(radius * 0.93, 0, 0),
  );
  part(
    new THREE.BoxGeometry(0.003, 0.001, 0.006),
    "#db9157",
    offset + 0.0175,
    false,
    new THREE.Vector3(radius * 0.78, 0, 0),
  );
}
