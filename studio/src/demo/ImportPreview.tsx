"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ImportGeometry, ObjectImportPreview } from "./types";
import styles from "./demo.module.css";

function geometry(asset: ImportGeometry) {
  const decode = (value: string) =>
    Uint8Array.from(atob(value), (c) => c.charCodeAt(0)).buffer;
  const result = new THREE.BufferGeometry();
  result.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(decode(asset.vertices)), 3),
  );
  result.setIndex(
    new THREE.BufferAttribute(new Uint32Array(decode(asset.faces)), 1),
  );
  if (asset.uv)
    result.setAttribute(
      "uv",
      new THREE.BufferAttribute(new Float32Array(decode(asset.uv)), 2),
    );
  if (asset.normals)
    result.setAttribute(
      "normal",
      new THREE.BufferAttribute(new Float32Array(decode(asset.normals)), 3),
    );
  else result.computeVertexNormals();
  return result;
}

export default function ImportPreview({
  preview,
}: {
  preview: ObjectImportPreview;
}) {
  const host = useRef<HTMLDivElement>(null);
  const hull = useRef<THREE.Mesh | null>(null);
  const [collision, setCollision] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    if (hull.current) hull.current.visible = collision;
  }, [collision]);
  useEffect(() => {
    const mount = host.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setError("Enable browser graphics acceleration to see the preview.");
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.domElement.setAttribute(
      "aria-label",
      "Imported object preview. Drag to rotate.",
    );
    renderer.domElement.setAttribute("role", "img");
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#e8e7e2");
    scene.add(new THREE.HemisphereLight(0xffffff, 0x6c7564, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5);
    light.position.set(1, -2, 3);
    scene.add(light);
    const texture = new THREE.TextureLoader().load(preview.texture_url);
    // Preview uses source OBJ UVs, before MuJoCo's V-coordinate conversion.
    texture.colorSpace = THREE.SRGBColorSpace;
    const visual = new THREE.Mesh(
      geometry(preview.visual),
      new THREE.MeshStandardMaterial({ map: texture, roughness: 0.8 }),
    );
    const proxy = new THREE.Mesh(
      geometry(preview.collision),
      new THREE.MeshBasicMaterial({
        color: 0x358765,
        wireframe: true,
        transparent: true,
        opacity: 0.35,
        depthTest: false,
      }),
    );
    proxy.visible = collision;
    hull.current = proxy;
    scene.add(visual, proxy);
    const size = Math.max(...(preview.object.dimensions ?? [0.12]));
    const camera = new THREE.PerspectiveCamera(38, 1, size / 100, size * 30);
    camera.up.set(0, 0, 1);
    camera.position.set(size * 1.6, -size * 2, size * 1.1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = size;
    controls.maxDistance = size * 8;
    const resize = new ResizeObserver(() => {
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      camera.aspect = mount.clientWidth / Math.max(1, mount.clientHeight);
      camera.updateProjectionMatrix();
    });
    resize.observe(mount);
    let frame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      hull.current = null;
      for (const mesh of [visual, proxy]) {
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
      texture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [preview]);
  return (
    <>
      <div ref={host} className={styles.importPreview} />
      {error && <p role="alert">{error}</p>}
      <label className={styles.collisionToggle}>
        <input
          type="checkbox"
          checked={collision}
          onChange={(e) => setCollision(e.target.checked)}
        />{" "}
        Show convex collision hull
      </label>
    </>
  );
}
