import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import type { RobotModel } from "../types";

type AddMesh = (
  body: number,
  geometry: THREE.BufferGeometry,
  color: string,
  position?: number[],
) => THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Presentation geometry follows existing link frames; it is not collision geometry. */
export function addResearchRobotDetails(model: RobotModel, add: AddMesh) {
  const builtIn =
    model.scene === "arm" ||
    model.scene === "dual_arm" ||
    model.scene === "dual_arm_head" ||
    model.scene === "mobile_dual_arm";
  if (!builtIn) return;
  const mobile = model.scene === "mobile_dual_arm";
  const graphite = "#22313d",
    metal = "#8799a6",
    shell = "#dce3e7",
    accent = "#ed8251";
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
  const cylinder = (
    body: number,
    radius: number,
    depth: number,
    color: string,
    position: number[],
  ) => {
    const mesh = add(
      body,
      new THREE.CylinderGeometry(radius, radius, depth, 48),
      color,
      position,
    );
    mesh.rotation.x = Math.PI / 2;
    return mesh;
  };
  model.names.forEach((name, body) => {
    if (mobile && (name.startsWith("left_") || name.startsWith("right_"))) {
      if (name.endsWith("waist")) {
        const socket = add(
          body,
          new THREE.SphereGeometry(1, 40, 28),
          "#e4e6e2",
          [0, 0, 0],
        );
        socket.scale.set(0.1, 0.105, 0.11);
      }
      const upper = name.endsWith("shoulder");
      const forearm = name.endsWith("elbow");
      if (upper || forearm) {
        const length = upper ? 0.32 : 0.28;
        const radius = upper ? 0.075 : 0.065;
        const sleeve = add(
          body,
          new THREE.CapsuleGeometry(radius, length - 2 * radius, 12, 32),
          "#e4e6e2",
          [length / 2, 0, 0],
        );
        sleeve.rotation.z = -Math.PI / 2;
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
    if (name.endsWith("pedestal") && !mobile) {
      box(body, [0.3, 0.3, 0.045], graphite, [0, 0, 0.024], 0.025);
      cylinder(body, 0.09, 0.34, shell, [0, 0, 0.225]);
      cylinder(body, 0.11, 0.045, graphite, [0, 0, 0.065]);
      cylinder(body, 0.097, 0.024, metal, [0, 0, 0.4]);
      cylinder(body, 0.098, 0.012, accent, [0, 0, 0.365]);
      for (const x of [-0.115, 0.115])
        for (const y of [-0.115, 0.115]) {
          const bolt = cylinder(body, 0.009, 0.009, metal, [x, y, 0.052]);
          bolt.material.metalness = 0.8;
        }
      for (let i = 0; i < 5; i++)
        box(body, [0.035, 0.004, 0.007], graphite, [
          0.078,
          -0.015 + i * 0.007,
          0.17,
        ]);
    }
    if (!mobile && (name.endsWith("shoulder") || name.endsWith("elbow"))) {
      const length = name.endsWith("shoulder") ? 0.32 : 0.28;
      box(body, [length * 0.57, 0.077, 0.013], graphite, [
        length * 0.52,
        0,
        0.043,
      ]);
      box(
        body,
        [length * 0.27, 0.079, 0.006],
        accent,
        [length * 0.43, 0, 0.053],
        0.002,
      );
      // Small service fasteners and a continuous lower cable cover.
      for (const x of [0.07, length - 0.055]) {
        cylinder(body, 0.004, 0.006, metal, [x, 0, 0.055]);
      }
      box(body, [length * 0.65, 0.024, 0.018], graphite, [
        length * 0.5,
        0,
        -0.048,
      ]);
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
        for (const side of [-1, 1])
          box(
            body,
            [0.06, 0.012, 0.032],
            "#67aaa4",
            [0.09, side * 0.025, 0],
            0.004,
          );
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
  } else {
    const root =
      model.scene === "dual_arm" || model.scene === "dual_arm_head"
        ? model.names.indexOf("world")
        : model.names.indexOf("pedestal");
    box(
      root,
      [
        0.38,
        model.scene === "dual_arm" || model.scene === "dual_arm_head"
          ? 1.12
          : 0.38,
        0.055,
      ],
      graphite,
      [-0.07, 0, 0.028],
      0.025,
    );
    box(root, [0.075, 0.075, 1.17], graphite, [-0.12, 0, 0.61], 0.009);
    box(root, [0.095, 0.2, 0.28], shell, [-0.15, 0, 0.71], 0.02);
    box(root, [0.006, 0.13, 0.16], graphite, [-0.098, 0, 0.72], 0.004);
    for (let i = 0; i < 6; i++)
      box(
        root,
        [0.008, 0.085, 0.006],
        metal,
        [-0.092, 0, 0.66 + i * 0.022],
        0.001,
      );
  }
  const head = model.names.indexOf("head");
  if (head >= 0 && mobile) {
    const shellMesh = add(head, new THREE.SphereGeometry(1, 40, 24), shell);
    shellMesh.scale.set(0.075, 0.105, 0.07);
    for (const y of [-0.032, 0.032]) {
      const lens = add(
        head,
        new THREE.CylinderGeometry(0.018, 0.018, 0.004, 32),
        graphite,
        [0.075, y, 0],
      );
      lens.rotation.z = Math.PI / 2;
    }
  }
  if (head >= 0 && !mobile) {
    box(head, [0.12, 0.19, 0.092], graphite, [0, 0, 0], 0.022);
    box(head, [0.008, 0.155, 0.052], "#101d29", [0.063, 0, 0], 0.009);
    box(head, [0.008, 0.08, 0.004], "#50b6ba", [0.069, 0, 0.033], 0.001);
    cylinder(head, 0.032, 0.07, metal, [0, 0, -0.065]);
  }
}
