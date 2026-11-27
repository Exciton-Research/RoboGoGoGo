"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import type { SceneData, TaskState } from "./types";
import styles from "./demo.module.css";

export default function TaskScene({
  sceneData,
  state,
  view,
}: {
  sceneData: SceneData;
  state: TaskState;
  view: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const overhead = useRef<HTMLCanvasElement>(null);
  const wrist = useRef<HTMLCanvasElement>(null);
  const latest = useRef(state);
  latest.current = state;
  const frame = useRef<() => void>(() => {});
  const [error, setError] = useState("");

  useEffect(() => {
    frame.current();
  }, [view]);
  useEffect(() => {
    const mount = host.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    let sensor: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
      sensor = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setError(
        "The 3D scene needs WebGL. Please enable browser graphics acceleration.",
      );
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    sensor.setSize(360, 216);
    sensor.toneMapping = THREE.ACESFilmicToneMapping;
    sensor.toneMappingExposure = 0.9;
    renderer.domElement.setAttribute(
      "aria-label",
      "Panda robot physically grasping and placing a block. Drag to orbit the workbench.",
    );
    renderer.domElement.setAttribute("role", "img");
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#e8e7e2");
    scene.fog = new THREE.Fog("#e8e7e2", 5, 13);
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    const environment = pmrem.fromScene(room, 0.04);
    scene.environment = environment.texture;
    scene.environmentIntensity = 0.5;
    room.dispose();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x7a817c, 0.85));
    const key = new THREE.DirectionalLight(0xfff5e9, 1.8);
    key.position.set(1.5, -1.5, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, {
      left: -1.7,
      right: 1.7,
      top: 2,
      bottom: -2,
      near: 0.1,
      far: 7,
    });
    key.shadow.normalBias = 0.001;
    key.shadow.bias = -0.00015;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xe6efff, 0.7);
    fill.position.set(-2, 2, 2.5);
    scene.add(fill);
    const meshAssets: Record<string, THREE.BufferGeometry> = {};
    const decode = (encoded: string) => {
      const raw = atob(encoded);
      const bytes = new Uint8Array(raw.length);
      for (let index = 0; index < raw.length; index++)
        bytes[index] = raw.charCodeAt(index);
      return bytes.buffer;
    };
    for (const [id, asset] of Object.entries(sceneData.meshes)) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(decode(asset.vertices)), 3),
      );
      geometry.setIndex(
        new THREE.BufferAttribute(new Uint32Array(decode(asset.faces)), 1),
      );
      geometry.computeVertexNormals();
      meshAssets[id] = geometry;
    }
    const objects = sceneData.geoms.map((geom) => {
      const [x, y, z] = geom.size;
      let geometry: THREE.BufferGeometry;
      if (geom.type === 7) geometry = meshAssets[String(geom.mesh)];
      else if (geom.type === 0) geometry = new THREE.PlaneGeometry(30, 30);
      else if (geom.type === 6) {
        const radius = geom.name === "block_geom" ? 0.002 : 0.004;
        geometry = new RoundedBoxGeometry(
          x * 2,
          y * 2,
          z * 2,
          2,
          Math.min(radius, x / 4, y / 4, z / 4),
        );
      } else if (geom.type === 5) {
        geometry = new THREE.CylinderGeometry(x, x, z * 2, 48);
        geometry.rotateX(Math.PI / 2);
      } else geometry = new THREE.SphereGeometry(x, 32, 24);
      const [r, g, b] = geom.rgba;
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b).convertSRGBToLinear(),
        roughness:
          geom.name === "block_geom"
            ? 0.5
            : geom.name === "worktop"
              ? 0.88
              : 0.38,
        metalness: geom.type === 7 ? 0.18 : 0.05,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = geom.type !== 0;
      mesh.receiveShadow = true;
      scene.add(mesh);
      return mesh;
    });
    const target = new THREE.Group();
    target.add(
      new THREE.Mesh(
        new THREE.PlaneGeometry(0.13, 0.13),
        new THREE.MeshStandardMaterial({
          color: 0x809b86,
          transparent: true,
          opacity: 0.18,
          roughness: 1,
        }),
      ),
    );
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x557963 });
    for (const sx of [-1, 1])
      for (const sy of [-1, 1]) {
        const corner = [
          new THREE.Vector3(sx * 0.041, sy * 0.065, 0),
          new THREE.Vector3(sx * 0.065, sy * 0.065, 0),
          new THREE.Vector3(sx * 0.065, sy * 0.041, 0),
        ];
        target.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(corner),
            lineMaterial,
          ),
        );
      }
    target.position.set(
      sceneData.state.target[0],
      sceneData.state.target[1],
      0.741,
    );
    scene.add(target);
    const camera = new THREE.PerspectiveCamera(37, 1, 0.01, 30);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.6;
    controls.maxDistance = 4;
    controls.maxPolarAngle = Math.PI * 0.49;
    const home = () => {
      camera.position.set(1.75, -1.8, 1.7);
      controls.target.set(0.28, 0, 1.0);
      controls.update();
    };
    frame.current = home;
    home();
    const topCamera = new THREE.PerspectiveCamera(42, 360 / 216, 0.01, 5);
    topCamera.up.set(0, 1, 0);
    topCamera.position.set(0.47, 0, 1.75);
    topCamera.lookAt(0.47, 0, 0.74);
    const wristCamera = new THREE.PerspectiveCamera(65, 360 / 216, 0.005, 4);
    const update = () => {
      const current = latest.current;
      current.poses.forEach((pose, index) => {
        if (!objects[index]) return;
        objects[index].position.set(pose[0], pose[1], pose[2]);
        objects[index].quaternion.set(pose[3], pose[4], pose[5], pose[6]);
      });
      for (const [name, camera] of [
        ["overhead", topCamera],
        ["wrist", wristCamera],
      ] as const) {
        const pose = current.cameras[name];
        camera.position.fromArray(pose.position);
        camera.quaternion.fromArray(pose.quaternion);
        camera.fov = pose.fovy;
        camera.updateProjectionMatrix();
      }
    };
    const resize = new ResizeObserver(() => {
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      camera.aspect = mount.clientWidth / Math.max(mount.clientHeight, 1);
      camera.updateProjectionMatrix();
    });
    resize.observe(mount);
    let request = 0,
      previousSensors = 0;
    const render = (time: number) => {
      update();
      controls.update();
      renderer.render(scene, camera);
      if (time - previousSensors > 120) {
        previousSensors = time;
        for (const [feed, viewCamera] of [
          [overhead.current, topCamera],
          [wrist.current, wristCamera],
        ] as const) {
          if (!feed) continue;
          sensor.render(scene, viewCamera);
          feed.getContext("2d")?.drawImage(sensor.domElement, 0, 0, 360, 216);
        }
      }
      request = window.requestAnimationFrame(render);
    };
    request = window.requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(request);
      resize.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      environment.dispose();
      pmrem.dispose();
      renderer.dispose();
      sensor.dispose();
      renderer.domElement.remove();
    };
  }, [sceneData]);
  return (
    <>
      <div ref={host} className={styles.canvas} />
      {error && (
        <p className={styles.sceneError} role="alert">
          {error}
        </p>
      )}
      <div className={styles.cameraStrip}>
        <div className={styles.cameraHeading}>
          <span className={styles.liveDot} />{" "}
          {state.replay?.rgb ? "RECORDED RGB" : "CAMERA PREVIEWS"}{" "}
          <span>
            {state.replay
              ? `Frame ${state.replay.frame}`
              : "Synchronized camera poses"}
          </span>
        </div>
        <div className={styles.feeds}>
          <figure>
            {state.replay?.rgb ? (
              <img
                src={`/api/demo/episodes/${state.replay.id}/frames/overhead/${state.replay.frame}`}
                alt="Recorded overhead RGB"
                width={360}
                height={216}
              />
            ) : (
              <canvas ref={overhead} width={360} height={216} />
            )}
            <figcaption>
              <span>01</span> Overhead
            </figcaption>
          </figure>
          <figure>
            {state.replay?.rgb ? (
              <img
                src={`/api/demo/episodes/${state.replay.id}/frames/wrist/${state.replay.frame}`}
                alt="Recorded wrist RGB"
                width={360}
                height={216}
              />
            ) : (
              <canvas ref={wrist} width={360} height={216} />
            )}
            <figcaption>
              <span>02</span> Wrist · virtual camera
            </figcaption>
          </figure>
        </div>
      </div>
    </>
  );
}
