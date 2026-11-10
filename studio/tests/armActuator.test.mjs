import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { addArmActuator } from "../src/rendering/armActuator.ts";

for (const [name, axis] of [
  ["left_waist", [0, 0, 1]],
  ["left_upper_roll", [1, 0, 0]],
  ["right_shoulder", [0, 1, 0]],
  ["left_elbow", [0, 1, 0]],
]) {
  test(`${name}: fixed housing and output stay coaxial during joint rotation`, () => {
    const rest = new THREE.Matrix4().makeRotationZ(0.4);
    const e = rest.elements;
    const origin = new THREE.Vector3(0.32, -0.1, 0.2);
    const model = {
      names: ["parent", name],
      parents: [-1, 0],
      axes: [[0, 0, 1], axis],
      joint_origins: [[0, 0, 0], origin.toArray()],
      joint_orientations: [
        null,
        [
          [e[0], e[4], e[8]],
          [e[1], e[5], e[9]],
          [e[2], e[6], e[10]],
        ],
      ],
    };
    const parent = new THREE.Group(),
      child = new THREE.Group();
    parent.position.set(1, 2, 3);
    parent.rotation.set(0.2, -0.3, 0.5);
    parent.add(child);
    child.position.copy(origin);
    const parts = [];
    addArmActuator(model, 1, (body, geometry, color, position) => {
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({ color }),
      );
      mesh.position.fromArray(position);
      (body === 0 ? parent : child).add(mesh);
      parts.push({ body, mesh });
      return mesh;
    });
    const housing = parts[0].mesh,
      flange = parts[2].mesh;
    const fixedMark = parts.at(-2).mesh,
      outputMark = parts.at(-1).mesh;
    let originalHousing, originalMark, originalOutput;
    for (const angle of [0, 0.8, -1.2]) {
      child.quaternion
        .setFromRotationMatrix(rest)
        .multiply(
          new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(...axis),
            angle,
          ),
        );
      parent.updateMatrixWorld(true);
      const hp = housing.getWorldPosition(new THREE.Vector3());
      const fp = flange.getWorldPosition(new THREE.Vector3());
      const axisWorld = new THREE.Vector3(0, 1, 0).applyQuaternion(
        housing.getWorldQuaternion(new THREE.Quaternion()),
      );
      assert.ok(fp.clone().sub(hp).cross(axisWorld).length() < 1e-9);
      const mark = fixedMark.getWorldPosition(new THREE.Vector3());
      const output = outputMark.getWorldPosition(new THREE.Vector3());
      if (angle === 0) {
        originalHousing = hp;
        originalMark = mark;
        originalOutput = output;
      } else {
        assert.ok(hp.distanceTo(originalHousing) < 1e-9);
        assert.ok(mark.distanceTo(originalMark) < 1e-9);
        assert.ok(output.distanceTo(originalOutput) > 0.005);
      }
    }
    for (const { mesh } of parts) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  });
}
