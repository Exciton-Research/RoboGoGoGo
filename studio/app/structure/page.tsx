"use client";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Frame = {
  time_s: number;
  complete: boolean;
  positions: number[][][];
  triangles: number[][];
  damage: number[][];
  fragments: { cells: number[] }[];
  gripper_centers: number[][];
  jaw_width_m: number[];
  maximum_damage: number;
  bend_m: number;
  broken_bonds: number;
  normal_force_n: number[][];
  peak_pressure_pa: number;
};
function Specimen({ frame }: { frame: Frame | null }) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef(frame);
  current.current = frame;
  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#101c29");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 20);
    camera.up.set(0, 0, 1);
    camera.position.set(0.48, -0.6, 0.45);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.06);
    scene.add(new THREE.HemisphereLight(0xe5f7ff, 0x456070, 3));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(1, -1, 2);
    scene.add(light);
    const grid = new THREE.GridHelper(1, 20, 0x456070, 0x263747);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    const objects = new THREE.Group();
    scene.add(objects);
    let previous: Frame | null = null;
    const clear = () => {
      for (const child of [...objects.children]) {
        const mesh = child as THREE.Mesh;
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
        objects.remove(mesh);
      }
    };
    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    renderer.setAnimationLoop(() => {
      const f = current.current;
      if (f && f !== previous) {
        clear();
        previous = f;
        f.positions.forEach((nodes, c) => {
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(nodes.flat(), 3),
          );
          geometry.setIndex(f.triangles.flat());
          geometry.computeVertexNormals();
          const damage = Math.max(
            ...(f.damage[c] ?? [0]),
            ...(f.damage[c - 1] ?? [0]),
          );
          const material = new THREE.MeshStandardMaterial({
            color: new THREE.Color("#73c9b6").lerp(
              new THREE.Color("#ee795a"),
              damage,
            ),
            roughness: 0.72,
            metalness: 0.05,
          });
          objects.add(new THREE.Mesh(geometry, material));
        });
        f.gripper_centers.forEach((center, i) =>
          [-1, 1].forEach((side) => {
            const mesh = new THREE.Mesh(
              new THREE.BoxGeometry(0.012, 0.07, 0.07),
              new THREE.MeshStandardMaterial({
                color: 0x8da2b9,
                metalness: 0.55,
                roughness: 0.3,
              }),
            );
            mesh.position.set(
              center[0] + side * (f.jaw_width_m[i] / 2 + 0.006),
              center[1],
              center[2],
            );
            objects.add(mesh);
          }),
        );
      }
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      renderer.setAnimationLoop(null);
      clear();
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      el.replaceChildren();
    };
  }, []);
  return (
    <div
      ref={host}
      style={{
        height: "62vh",
        minHeight: 360,
        borderRadius: 16,
        overflow: "hidden",
      }}
    />
  );
}
export default function StructurePage() {
  const [frame, setFrame] = useState<Frame | null>(null),
    [material, setMaterial] = useState("soft_foam"),
    [recipe, setRecipe] = useState("bend"),
    [running, setRunning] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const request = async (path: string, body?: object) => {
    const response = await fetch(`/api/structure/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.detail ?? "Simulation request failed");
    return data as Frame;
  };
  const reset = async () => {
    setRunning(false);
    setBusy(true);
    setError("");
    try {
      setFrame(await request("reset", { material, recipe }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const data = await request("step");
        if (cancelled) return;
        setFrame(data);
        if (data.complete) setRunning(false);
        else timer = setTimeout(tick, 0);
      } catch (e) {
        if (!cancelled) {
          setError(String(e));
          setRunning(false);
        }
      }
    };
    let timer = setTimeout(tick, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [running]);
  return (
    <main
      style={{
        maxWidth: 1280,
        margin: "auto",
        padding: 32,
        color: "#dfeaf4",
        background: "#101925",
        minHeight: "100vh",
      }}
    >
      <a href="/" style={{ color: "#82d9c5" }}>
        ← Robotics Studio
      </a>
      <h1>Structural Lab</h1>
      <p>Contact-driven bending, tearing and fracture · Python / JAX</p>
      <p style={{ color: "#9bafc2" }}>
        A destructive specimen bench. Cracks follow cell interfaces; tools
        follow prescribed trajectories. This model is separate from the arm
        handover controller.
      </p>
      <div
        style={{ display: "flex", gap: 12, flexWrap: "wrap", margin: "24px 0" }}
      >
        <select
          aria-label="Material"
          value={material}
          disabled={running || busy}
          onChange={(e) => setMaterial(e.target.value)}
        >
          <option value="soft_foam">Soft foam</option>
          <option value="soft_elastomer">Elastomer</option>
          <option value="fragile_shell">Fragile specimen</option>
        </select>
        <select
          aria-label="Experiment"
          value={recipe}
          disabled={running || busy}
          onChange={(e) => setRecipe(e.target.value)}
        >
          <option value="bend">Cantilever lift</option>
          <option value="tear">Two-tool tear</option>
          <option value="crush">Overforce crush</option>
        </select>
        <button onClick={reset} disabled={running || busy}>
          {busy ? "Preparing…" : "Reset specimen"}
        </button>
        <button
          disabled={!frame || busy || frame.complete}
          onClick={() => setRunning(!running)}
        >
          {running ? "Pause" : "Run"}
        </button>
        <a style={{ color: "#82d9c5" }} href="/api/structure/export">
          Export trajectory
        </a>
      </div>
      {error && <p role="alert">{error}</p>}
      <Specimen frame={frame} />
      {frame && (
        <div
          style={{
            display: "flex",
            gap: 32,
            flexWrap: "wrap",
            padding: "24px 0",
          }}
        >
          <span>
            Time <b>{frame.time_s.toFixed(2)} s</b>
          </span>
          <span>
            Pieces <b>{frame.fragments.length}</b>
          </span>
          <span>
            Broken bonds <b>{frame.broken_bonds}</b>
          </span>
          <span>
            Damage <b>{(100 * frame.maximum_damage).toFixed(0)}%</b>
          </span>
          <span>
            Centerline offset <b>{(1000 * frame.bend_m).toFixed(2)} mm</b>
          </span>
          <span>
            Peak pressure{" "}
            <b>{(frame.peak_pressure_pa / 1000).toFixed(1)} kPa</b>
          </span>
          <span>
            Jaw loads{" "}
            <b>
              {frame.normal_force_n
                .flat()
                .map((n) => n.toFixed(1))
                .join(" / ")}{" "}
              N
            </b>
          </span>
        </div>
      )}
      <p>
        Drag to orbit · scroll to zoom. Red marks failed or weakened interfaces.
        Reset applies the selected material and experiment. First run compiles
        the native solver.
      </p>
    </main>
  );
}
