import * as THREE from "three";
import type { RobotModel, Snapshot } from "../types";

/** Gripper/object meshes follow backend state; no browser-side attachment logic. */
export function createManipulationVisuals(
  scene: THREE.Scene,
  bodies: THREE.Group[],
  model: RobotModel,
) {
  const orange = new THREE.MeshStandardMaterial({
    color: 0xee8647,
    roughness: 0.48,
    metalness: 0.16,
    vertexColors: true,
  });
  const graphite = new THREE.MeshStandardMaterial({
    color: 0x263744,
    roughness: 0.42,
    metalness: 0.6,
  });
  const pad = new THREE.MeshStandardMaterial({
    color: 0x131b22,
    roughness: 0.9,
  });
  // Axial rings resolve local dents; a two-ring cylinder cannot show them.
  const profile: THREE.Vector2[] = [];
  for (let i = 0; i <= 8; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 16;
    profile.push(
      new THREE.Vector2(0.03 * Math.cos(a), -0.12 + 0.03 * Math.sin(a)),
    );
  }
  for (let i = 1; i < 32; i++)
    profile.push(new THREE.Vector2(0.03, -0.12 + (0.24 * i) / 32));
  for (let i = 0; i <= 8; i++) {
    const a = (i * Math.PI) / 16;
    profile.push(
      new THREE.Vector2(0.03 * Math.cos(a), 0.12 + 0.03 * Math.sin(a)),
    );
  }
  const geometry = new THREE.LatheGeometry(profile, 48);
  const original = new Float32Array(geometry.getAttribute("position").array);
  geometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(
      new Float32Array(original.length).fill(1),
      3,
    ),
  );
  const object = new THREE.Mesh(geometry, orange);
  const healthyColor = new THREE.Color();
  const damageColor = new THREE.Color("#b52b27");
  const vertexColor = new THREE.Color();
  object.castShadow = true;
  object.receiveShadow = true;
  object.visible = false;
  scene.add(object);
  const grippers = model.arms.map((arm) => {
    const group = new THREE.Group();
    bodies[arm.body].add(group);
    const palm = new THREE.Mesh(
      new THREE.BoxGeometry(0.025, 0.12, 0.035),
      graphite,
    );
    palm.position.x = 0.008;
    group.add(palm);
    const fingers = [-1, 1].map((side) => {
      const finger = new THREE.Group();
      group.add(finger);
      const shell = new THREE.Mesh(
        new THREE.BoxGeometry(0.21, 0.01, 0.016),
        graphite,
      );
      shell.position.x = 0.125;
      shell.castShadow = true;
      finger.add(shell);
      const contact = new THREE.Mesh(
        new THREE.BoxGeometry(0.045, 0.004, 0.018),
        pad,
      );
      contact.position.set(0.207, -side * 0.005, 0);
      finger.add(contact);
      return { finger, side };
    });
    group.visible = false;
    return { arm, group, fingers };
  });
  const structuralGroup = new THREE.Group();
  scene.add(structuralGroup);
  let structuralMeshes: THREE.Mesh[] = [];
  const matrix = new THREE.Matrix4();
  return (snapshot: Snapshot) => {
    const state = snapshot.manipulation;
    const structural = state?.object.structure;
    object.visible = !!state && !structural;
    structuralGroup.visible = !!structural;
    if (structural) {
      if (structuralMeshes.length !== structural.positions.length) {
        for (const mesh of structuralMeshes) {
          mesh.geometry.dispose();
          (mesh.material as THREE.Material).dispose();
          structuralGroup.remove(mesh);
        }
        structuralMeshes = structural.positions.map(() => {
          const g = new THREE.BufferGeometry();
          g.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(new Float32Array(24), 3),
          );
          g.setIndex(structural.triangles.flat());
          const mesh = new THREE.Mesh(
            g,
            new THREE.MeshStandardMaterial({
              color: state!.object.material.parameters?.color ?? "#77b8ad",
              roughness: 0.7,
            }),
          );
          mesh.castShadow = mesh.receiveShadow = true;
          structuralGroup.add(mesh);
          return mesh;
        });
      }
      structuralMeshes.forEach((mesh, i) => {
        const p = mesh.geometry.getAttribute(
          "position",
        ) as THREE.BufferAttribute;
        structural.positions[i].forEach((v, k) =>
          p.setXYZ(k, v[0], v[1], v[2]),
        );
        p.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
        mesh.geometry.computeBoundingSphere();
        const damage = Math.max(
          ...(structural.damage[i] ?? [0]),
          ...(structural.damage[i - 1] ?? [0]),
        );
        (mesh.material as THREE.MeshStandardMaterial).color
          .set(state!.object.material.parameters?.color ?? "#77b8ad")
          .lerp(damageColor, damage);
      });
    }
    if (state) {
      object.position.set(...state.object.position);
      const material = state.object.material;
      const dents = material?.indentation_m;
      const damage = material?.damage;
      healthyColor.set(material?.parameters?.color ?? "#ee8647");
      orange.color.set("#ffffff");
      const positions = geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;
      const colors = geometry.getAttribute("color") as THREE.BufferAttribute;
      for (let i = 0; i < positions.count; i++) {
        const x = original[3 * i],
          y = original[3 * i + 1],
          z = original[3 * i + 2];
        const row = Math.max(0, Math.min(31, ((y + 0.15) / 0.3) * 32 - 0.5));
        const angle = (Math.atan2(z, x) + 2 * Math.PI) % (2 * Math.PI);
        const col = (angle / (2 * Math.PI)) * 24 - 0.5;
        const iy = Math.floor(row),
          ia = Math.floor(col);
        const wy = row - iy,
          wa = col - ia;
        const sample = (field?: number[][]) => {
          if (!field) return 0;
          const a = (ia + 24) % 24,
            b = (ia + 25) % 24,
            next = Math.min(31, iy + 1);
          return (
            (1 - wy) * ((1 - wa) * field[iy][a] + wa * field[iy][b]) +
            wy * ((1 - wa) * field[next][a] + wa * field[next][b])
          );
        };
        const depth = Math.min(0.024, sample(dents));
        const centerY = Math.max(-0.12, Math.min(0.12, y));
        const length = Math.hypot(x, y - centerY, z);
        const factor = length > 1e-9 ? 1 - depth / length : 1;
        positions.setXYZ(
          i,
          x * factor,
          centerY + (y - centerY) * factor,
          z * factor,
        );
        vertexColor.copy(healthyColor).lerp(damageColor, sample(damage));
        colors.setXYZ(i, vertexColor.r, vertexColor.g, vertexColor.b);
      }
      positions.needsUpdate = colors.needsUpdate = true;
      geometry.computeVertexNormals();
      geometry.computeBoundingSphere();
      const r = state.object.rotation;
      matrix.set(
        r[0][0],
        r[0][1],
        r[0][2],
        0,
        r[1][0],
        r[1][1],
        r[1][2],
        0,
        r[2][0],
        r[2][1],
        r[2][2],
        0,
        0,
        0,
        0,
        1,
      );
      object.quaternion.setFromRotationMatrix(matrix);
    }
    for (const { arm, group, fingers } of grippers) {
      const gripper = state?.grippers[arm.name];
      group.visible = !!gripper;
      if (gripper)
        for (const { finger, side } of fingers)
          finger.position.y = side * (gripper.width / 2 + 0.007);
    }
  };
}
