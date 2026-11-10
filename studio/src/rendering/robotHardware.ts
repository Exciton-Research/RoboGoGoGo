import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import type { CameraConfig, RobotModel } from "../types";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Illustrative hardware details; all parts inherit actual kinematic link frames. */
export function addRobotHardware(
  model: RobotModel,
  cameras: CameraConfig[],
  add: AddMesh,
) {
  const axisY = new THREE.Vector3(0, 1, 0);
  const metal = "#8d989c",
    dark = "#263334",
    rubber = "#202828";
  for (const joint of model.joints) {
    if (
      !/(waist|upper_roll|shoulder|elbow|wrist_pitch|torso_lower|torso_upper|_steer)$/.test(
        joint.name,
      )
    )
      continue;
    const axis = new THREE.Vector3(...model.axes[joint.body]).normalize();
    const orientation = new THREE.Quaternion().setFromUnitVectors(axisY, axis);
    const wrist = joint.name.includes("wrist"),
      steering = joint.name.endsWith("steer");
    const shoulderAxis = /_(waist|upper_roll|shoulder)$/.test(joint.name);
    const radius = shoulderAxis
      ? 0.033
      : wrist
        ? 0.035
        : steering
          ? 0.043
          : 0.057;
    const offset = shoulderAxis
      ? 0.082
      : wrist
        ? 0.038
        : steering
          ? 0.045
          : 0.073;
    // Three exposed actuator endbells follow the actual yaw, roll and pitch frames.
    // These are concept housings, not selected stepper or servo specifications.
    for (const side of [-1, 1]) {
      const center = axis.clone().multiplyScalar(side * offset);
      const cap = add(
        joint.body,
        new THREE.CylinderGeometry(
          radius,
          radius,
          shoulderAxis ? 0.018 : 0.009,
          40,
        ),
        dark,
        center.toArray(),
      );
      cap.quaternion.copy(orientation);
      cap.material.roughness = 0.28;
      const ring = add(
        joint.body,
        new THREE.TorusGeometry(radius * 0.82, 0.0025, 8, 48),
        metal,
        center
          .clone()
          .addScaledVector(axis, side * 0.005)
          .toArray(),
      );
      ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      const disk = add(
        joint.body,
        new THREE.CylinderGeometry(radius * 0.32, radius * 0.32, 0.011, 24),
        metal,
        center.toArray(),
      );
      disk.quaternion.copy(orientation);
      disk.material.metalness = 0.8;
      for (let i = 0; i < 6; i++) {
        const angle = (i * Math.PI) / 3;
        const radial = new THREE.Vector3(
          Math.cos(angle) * radius * 0.63,
          side * 0.006,
          Math.sin(angle) * radius * 0.63,
        ).applyQuaternion(orientation);
        const bolt = add(
          joint.body,
          new THREE.CylinderGeometry(0.0025, 0.0025, 0.002, 6),
          metal,
          center.clone().add(radial).toArray(),
        );
        bolt.quaternion.copy(orientation);
      }
    }
  }
  model.names.forEach((name, body) => {
    if (name.endsWith("_wheel")) {
      for (let i = 0; i < 24; i++) {
        const angle = (i * Math.PI) / 12;
        const tread = add(
          body,
          new THREE.BoxGeometry(0.018, 0.074, 0.005),
          rubber,
          [0.14 * Math.sin(angle), 0, 0.14 * Math.cos(angle)],
        );
        tread.rotation.y = angle;
        tread.material.roughness = 0.95;
      }
      for (const side of [-1, 1]) {
        const hub = add(
          body,
          new THREE.CylinderGeometry(0.039, 0.039, 0.013, 32),
          dark,
          [0, side * 0.052, 0],
        );
        hub.material.metalness = 0.65;
      }
    }
    if (name === "base_link") {
      for (const side of [-1, 1]) {
        for (let i = 0; i < 9; i++) {
          add(body, new THREE.BoxGeometry(0.006, 0.002, 0.035), dark, [
            -0.15 + i * 0.018,
            side * 0.371,
            -0.005,
          ]);
        }
        const rail = add(
          body,
          new RoundedBoxGeometry(0.48, 0.012, 0.018, 2, 0.004),
          dark,
          [0, side * 0.375, 0.045],
        );
        rail.material.roughness = 0.6;
      }
    }
    if (name.endsWith("shoulder") || name.endsWith("elbow")) {
      const length = name.endsWith("shoulder") ? 0.32 : 0.28;
      const panel = add(
        body,
        new RoundedBoxGeometry(length * 0.45, 0.04, 0.008, 3, 0.004),
        "#c1c9c5",
        [length * 0.5, 0, -0.063],
      );
      panel.material.roughness = 0.65;
    }
  });
  // Camera optical origins and rotations use the same episode calibration as RGB capture.
  for (const camera of cameras) {
    const body = camera.mount ? model.names.indexOf(camera.mount) : -1;
    if (body < 0) continue;
    const origin = new THREE.Vector3(...camera.position);
    const orientation = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(
        origin,
        new THREE.Vector3(...camera.look_at),
        new THREE.Vector3(...camera.up),
      ),
    );
    const place = (
      geometry: THREE.BufferGeometry,
      color: string,
      z: number,
    ) => {
      const position = origin
        .clone()
        .add(new THREE.Vector3(0, 0, z).applyQuaternion(orientation));
      const mesh = add(body, geometry, color, position.toArray());
      mesh.quaternion.copy(orientation);
      return mesh;
    };
    place(new RoundedBoxGeometry(0.035, 0.028, 0.026, 3, 0.004), dark, 0.018);
    const barrel = place(
      new THREE.CylinderGeometry(0.012, 0.013, 0.012, 32),
      "#121e22",
      0.007,
    );
    barrel.rotateX(Math.PI / 2);
    const bezel = place(
      new THREE.TorusGeometry(0.011, 0.0015, 8, 32),
      metal,
      0.002,
    );
    bezel.material.metalness = 0.85;
    const glass = place(new THREE.CircleGeometry(0.0095, 32), "#204f62", 0.001);
    glass.rotateY(Math.PI);
    glass.material.roughness = 0.08;
    glass.material.metalness = 0.75;
    if (camera.mount?.endsWith("tool")) {
      add(body, new RoundedBoxGeometry(0.018, 0.022, 0.052, 2, 0.003), metal, [
        camera.position[0],
        camera.position[1],
        camera.position[2] - 0.04,
      ]);
    }
  }
}
