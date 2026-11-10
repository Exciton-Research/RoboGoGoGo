import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import type { RobotModel } from "../types";
import { armShell } from "./armShell";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Presentation geometry follows existing link frames; it is not collision geometry. */
export function addResearchRobotDetails(model: RobotModel, add: AddMesh) {
  const mobile = model.scene === "mobile_dual_arm";
  const graphite = "#22313d",
    metal = "#8799a6",
    shell = "#dce3e7";
  const box = (
    body: number,
    size: number[],
    color: string,
    position: number[],
    radius = 0.012,
  ) =>
    add(
      body,
      new RoundedBoxGeometry(
        size[0],
        size[1],
        size[2],
        3,
        Math.min(radius, ...size.map((v) => v / 4)),
      ),
      color,
      position,
    );
  model.names.forEach((name, body) => {
    if (mobile && (name.startsWith("left_") || name.startsWith("right_"))) {
      if (name.endsWith("waist")) {
        const socket = add(
          body,
          new THREE.SphereGeometry(1, 40, 28),
          "#e4e6e2",
          [0, 0, 0],
        );
        socket.userData.removableCover = true;
        socket.scale.set(0.166, 0.166, 0.166);
      }
      const upper = name.endsWith("shoulder");
      const forearm = name.endsWith("elbow");
      if (upper || forearm) {
        const length = upper ? 0.32 : 0.28;
        const radius = upper ? 0.075 : 0.065;
        // Two removable shells around a continuous dark structural carrier.
        const carrier = add(
          body,
          new THREE.CylinderGeometry(radius * 0.53, radius * 0.53, length, 40),
          graphite,
          [length / 2, 0, 0],
        );
        carrier.userData.removableCover = true;
        carrier.rotation.z = -Math.PI / 2;
        for (const half of [0, 1]) {
          const sleeve = add(body, armShell(length, radius, half), shell);
          sleeve.userData.removableCover = true;
          sleeve.rotation.z = -Math.PI / 2;
          sleeve.material.roughness = 0.46;
          sleeve.material.metalness = 0.08;
        }
        for (const x of [0.025, length - 0.03]) {
          const collar = add(
            body,
            new THREE.CylinderGeometry(radius * 0.6, radius * 0.6, 0.009, 48),
            metal,
            [x, 0, 0],
          );
          collar.rotation.z = -Math.PI / 2;
          collar.material.metalness = 0.75;
        }
        if (forearm) {
          add(
            body,
            new THREE.SphereGeometry(0.068, 32, 24),
            "#aeb6b3",
            [0, 0, 0],
          );
          const cap = add(
            body,
            new THREE.CylinderGeometry(0.057, 0.057, 0.14, 40),
            "#e4e6e2",
            [0, 0, 0],
          );
          cap.rotation.y = 0;
        }
      }
      for (const [suffix, length, radius] of [
        ["wrist_roll", 0.04, 0.043],
        ["wrist_pitch", 0.03, 0.043],
        ["tool_roll", 0.04, 0.04],
      ] as const) {
        if (name.endsWith(suffix)) {
          const cuff = add(
            body,
            new THREE.CylinderGeometry(radius, radius, length, 32),
            suffix === "wrist_pitch" ? "#aeb6b3" : "#e4e6e2",
            [length / 2, 0, 0],
          );
          cuff.rotation.z = Math.PI / 2;
        }
      }
    }

    if (name.endsWith("tool")) {
      const flange = add(
        body,
        new THREE.CylinderGeometry(0.035, 0.035, 0.018, 40),
        graphite,
        [-0.01, 0, 0],
      );
      flange.rotation.z = Math.PI / 2;
      const ring = add(
        body,
        new THREE.TorusGeometry(0.026, 0.003, 8, 32),
        metal,
        [0.001, 0, 0],
      );
      ring.rotation.y = Math.PI / 2;
      // Concept fingers are presentation geometry, not a contact/gripper model.
      if (mobile) {
        box(body, [0.055, 0.072, 0.045], graphite, [0.035, 0, 0]);
        for (const side of [-1, 1]) {
          box(
            body,
            [0.05, 0.005, 0.026],
            "#263334",
            [0.092, side * 0.017, 0],
            0.002,
          );
          box(
            body,
            [0.06, 0.012, 0.032],
            "#8799a6",
            [0.09, side * 0.025, 0],
            0.004,
          );
        }
      }
    }
  });
  if (mobile) {
    const base = model.names.indexOf("base_link");
    for (const side of [-1, 1])
      box(
        base,
        [0.014, 0.1, 0.035],
        "#67aaa4",
        [0.385, side * 0.27, -0.01],
        0.004,
      );
    for (const [name, radius] of [
      ["torso_lower", 0.075],
      ["torso_upper", 0.065],
    ] as const) {
      const body = model.names.indexOf(name);
      const shellMesh = add(
        body,
        new THREE.CapsuleGeometry(radius, 0.44 - 2 * radius, 8, 24),
        shell,
        [0, 0, 0.22],
      );
      shellMesh.rotation.x = Math.PI / 2;
    }
    const chest = add(
      model.names.indexOf("torso"),
      new THREE.SphereGeometry(1, 40, 24),
      shell,
      [0, 0, 0.14],
    );
    chest.scale.set(0.15, 0.26, 0.19);
    const torso = model.names.indexOf("torso");
    // Fixed shoulder bearing carriers bridge the chest to each yaw assembly.
    for (const side of [-1, 1]) {
      add(torso, new THREE.CylinderGeometry(0.06, 0.06, 0.105, 40), graphite, [
        0,
        side * 0.285,
        0.22,
      ]);
      add(torso, new THREE.CylinderGeometry(0.064, 0.064, 0.012, 40), metal, [
        0,
        side * 0.32,
        0.22,
      ]);
    }

    const neck = add(
      torso,
      new THREE.CylinderGeometry(0.044, 0.052, 0.07, 40),
      graphite,
      [0, 0, 0.345],
    );
    neck.rotation.x = Math.PI / 2;

    box(
      model.names.indexOf("torso"),
      [0.012, 0.12, 0.018],
      "#67aaa4",
      [0.15, 0, 0.19],
      0.004,
    );
    model.names.forEach((name, body) => {
      if (!name.endsWith("_wheel")) return;
      for (const y of [-0.043, 0.043]) {
        const hub = add(
          body,
          new THREE.CylinderGeometry(0.085, 0.085, 0.008, 32),
          metal,
          [0, y, 0],
        );
        hub.rotation.y = 0;
        box(body, [0.014, 0.01, 0.19], graphite, [0, y * 1.1, 0], 0.004);
      }
    });
  }
  const head = model.names.indexOf("head");
  if (head >= 0 && mobile) {
    const shellMesh = add(head, new THREE.SphereGeometry(1, 40, 24), shell);
    shellMesh.scale.set(0.095, 0.13, 0.09);
    // Stereo lens origins remain at x=0.105, ahead of this recessed face plate.
    box(head, [0.016, 0.17, 0.045], graphite, [0.085, 0, 0], 0.01);
    for (const side of [-1, 1]) {
      const ear = add(
        head,
        new THREE.CylinderGeometry(0.031, 0.031, 0.009, 32),
        metal,
        [0, side * 0.125, 0],
      );
      ear.material.metalness = 0.7;
    }
  }
}
