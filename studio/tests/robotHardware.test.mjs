import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { addRobotHardware } from "../src/rendering/robotHardware.ts";

function render(model, cameras = []) {
  const parts = [];
  addRobotHardware(
    model,
    cameras,
    (body, geometry, color, position = [0, 0, 0]) => {
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({ color }),
      );
      mesh.position.fromArray(position);
      parts.push({ body, mesh });
      return mesh;
    },
  );
  return parts;
}

test("catalog arm motors do not acquire duplicate generic housings", () => {
  const names = [
    "waist",
    "upper_roll",
    "shoulder",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "tool_roll",
  ].flatMap((joint) => ["left_" + joint, "right_" + joint]);
  const parts = render({
    names,
    joints: names.map((name, body) => ({ name, body })),
    axes: names.map(() => [0, 1, 0]),
  });
  assert.equal(parts.length, 0);
});

test("camera lens faces the configured optical look-at direction", () => {
  const position = new THREE.Vector3(0.1, -0.2, 0.3);
  const target = new THREE.Vector3(0.8, 0.4, 0.5);
  const parts = render({ names: ["head"], joints: [] }, [
    {
      mount: "head",
      position: position.toArray(),
      look_at: target.toArray(),
      up: [0, 0, 1],
    },
  ]);
  try {
    const glass = parts.find(
      ({ mesh }) => mesh.geometry.type === "CircleGeometry",
    );
    assert.ok(glass);
    assert.equal(glass.body, 0);
    const forward = target.clone().sub(position).normalize();
    const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(
      glass.mesh.quaternion,
    );
    assert.ok(normal.distanceTo(forward) < 1e-9);
  } finally {
    for (const { mesh } of parts) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  }
});
