import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
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
    segment(p.body, a, b, p.radius, p.shoulder_axis ? "#263e60" : "#34414a");
    const direction = b.clone().sub(a).normalize();
    for (const end of [a, b])
      segment(
        p.body,
        end.clone().addScaledVector(direction, -0.002),
        end.clone().addScaledVector(direction, 0.002),
        p.radius,
        "#a1aeb5",
      );
    // All output connections come from backend parts on the rotating link.
  }
  for (const p of model.assembly_parts ?? []) {
    const fitting = segment(
      p.body,
      new THREE.Vector3(...p.start),
      new THREE.Vector3(...p.end),
      p.radius,
      p.kind === "connector" || p.kind === "wire" ? "#232b30" : "#9aa8b1",
    );
    if (p.render_style === "shoulder_yoke") {
      fitting.geometry.dispose();
      const length = new THREE.Vector3(...p.end).distanceTo(
        new THREE.Vector3(...p.start),
      );
      fitting.geometry = new RoundedBoxGeometry(
        p.section_m![0],
        length,
        p.section_m![1],
        3,
        0.002,
      );
      fitting.material.color.set("#e4e6e2");
      fitting.material.roughness = 0.34;
      fitting.material.metalness = 0.22;
    }
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
  for (const cover of model.motor_covers ?? []) {
    const a = new THREE.Vector3(...cover.start);
    const b = new THREE.Vector3(...cover.end);
    const direction = b.clone().sub(a).normalize();
    // Two open-ended annular halves: the motor shaft stays accessible and the
    // cover can be separated radially without threading it over the arm.
    const gap = cover.seam_gap_m / (2 * cover.inner_radius);
    for (const half of [0, 1]) {
      const start = half * Math.PI + gap;
      const end = (half + 1) * Math.PI - gap;
      const section = new THREE.Shape();
      section.absarc(0, 0, cover.radius, start, end, false);
      section.lineTo(
        cover.inner_radius * Math.cos(end),
        cover.inner_radius * Math.sin(end),
      );
      section.absarc(0, 0, cover.inner_radius, end, start, true);
      section.closePath();
      const length = a.distanceTo(b);
      const shoulder = /_(waist|upper_roll|shoulder)_split_cover$/.test(
        cover.name,
      );
      const zones = shoulder
        ? ([
            [0, 0.012, "#e4e6e2"],
            [0.012, length - 0.024, "#263e60"],
            [length - 0.012, 0.012, "#e4e6e2"],
          ] as const)
        : ([[0, length, "#e4e6e2"]] as const);
      for (const [offset, depth, color] of zones) {
        const skin = add(
          cover.body,
          new THREE.ExtrudeGeometry(section, {
            depth,
            bevelEnabled: false,
            curveSegments: 40,
          }),
          color,
          a.clone().addScaledVector(direction, offset).toArray(),
        );
        skin.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          direction,
        );
        skin.userData.removableCover = true;
        skin.material.roughness = color === "#263e60" ? 0.48 : 0.34;
        skin.material.metalness = color === "#263e60" ? 0.5 : 0.15;
      }
    }
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
