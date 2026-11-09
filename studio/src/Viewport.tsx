import { createManipulationVisuals } from "./rendering/manipulation";
import { captureDepth } from "./rendering/depthCapture";
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { addResearchRobotDetails } from "./rendering/robotAppearance";
import type { Snapshot, Vec3 } from "./types";

export interface ViewportHandle {
  view: (view: "perspective" | "top" | "front") => void;
  capture: (snapshot: Snapshot) => {
    images: Record<string, string>;
    ground_truth_depth: Record<string, string>;
  };
}
interface Props {
  snapshot: Snapshot;
  trail: Vec3[];
  selected: number | null;
  onSelect: (body: number) => void;
  showGrid: boolean;
  showAxes: boolean;
}

export const Viewport = forwardRef<ViewportHandle, Props>(
  function Viewport(props, forwardedRef) {
    const mount = useRef<HTMLDivElement>(null);
    const sensorMount = useRef<HTMLDivElement>(null);
    const captureFrame = useRef<
      (snapshot: Snapshot) => {
        images: Record<string, string>;
        ground_truth_depth: Record<string, string>;
      }
    >(() => {
      throw new Error("Camera is still initializing");
    });
    const current = useRef(props);
    current.current = props;
    const cameraControl = useRef<
      (view: "perspective" | "top" | "front") => void
    >(() => {});
    const [failure, setFailure] = useState("");
    const [previews, setPreviews] = useState(true);
    useImperativeHandle(
      forwardedRef,
      () => ({
        view: (view) => cameraControl.current(view),
        capture: (snapshot) => captureFrame.current(snapshot),
      }),
      [],
    );

    useEffect(() => {
      const host = mount.current;
      if (!host) return;
      const { model } = current.current.snapshot;
      let renderer: THREE.WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      } catch {
        setFailure(
          "WebGL is unavailable. Enable hardware acceleration to use the 3D viewport. The Python controls remain available.",
        );
        return;
      }
      setFailure("");
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      renderer.domElement.setAttribute(
        "aria-label",
        "Interactive 3D robot scene. Drag to orbit, scroll to zoom, click a link to inspect.",
      );
      renderer.domElement.setAttribute("role", "img");
      host.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#e9ece7");
      scene.fog = new THREE.Fog("#e9ece7", 5, 13);
      const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
      camera.up.set(0, 0, 1);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.09;
      controls.minDistance = 0.3;
      controls.maxDistance = 10;
      controls.maxPolarAngle = Math.PI * 0.52;
      const defaultTarget =
        model.scene === "drop"
          ? new THREE.Vector3(0, 0, 0.3)
          : new THREE.Vector3(0.3, 0, 0.62);
      const home = (view: "perspective" | "top" | "front") => {
        controls.target.copy(defaultTarget);
        if (view === "top")
          camera.position
            .copy(defaultTarget)
            .add(new THREE.Vector3(0, -0.001, 2.7));
        else if (view === "front")
          camera.position
            .copy(defaultTarget)
            .add(new THREE.Vector3(0, -2.7, 0.12));
        else
          camera.position
            .copy(defaultTarget)
            .add(new THREE.Vector3(1.65, -2.2, 1.5));
        controls.update();
      };
      cameraControl.current = home;
      home("perspective");
      scene.add(new THREE.HemisphereLight("#ffffff", "#8f9a91", 1.6));
      const sun = new THREE.DirectionalLight("#fff5e5", 2.8);
      sun.position.set(2, -3, 5);
      sun.castShadow = true;
      sun.shadow.mapSize.set(2048, 2048);
      Object.assign(sun.shadow.camera, {
        left: -2,
        right: 2,
        top: 2,
        bottom: -2,
        near: 0.1,
        far: 12,
      });
      sun.shadow.normalBias = 0.015;
      scene.add(sun);
      const fill = new THREE.DirectionalLight("#d8e5ff", 1.3);
      fill.position.set(-2, 1, 3);
      scene.add(fill);
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(100, 100),
        new THREE.MeshStandardMaterial({ color: "#e5e9e2", roughness: 0.92 }),
      );
      floor.position.z = -0.004;
      floor.receiveShadow = true;
      scene.add(floor);
      for (const obstacle of current.current.snapshot.coordination?.obstacles ??
        []) {
        const mesh = new THREE.Mesh(
          obstacle.kind === "box"
            ? new THREE.BoxGeometry(...obstacle.size)
            : new THREE.SphereGeometry(obstacle.radius, 32, 24),
          new THREE.MeshStandardMaterial({
            color: obstacle.kind === "box" ? 0x657780 : 0xd97b3f,
            roughness: 0.65,
            metalness: 0.12,
          }),
        );
        mesh.position.set(...obstacle.center);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
      }
      const grid = new THREE.GridHelper(12, 120, "#a3afa3", "#c8d0c5");
      grid.rotation.x = Math.PI / 2;
      grid.position.z = -0.002;
      (grid.material as THREE.Material).transparent = true;
      (grid.material as THREE.Material).opacity = 0.26;
      scene.add(grid);
      const origin = new THREE.AxesHelper(0.22);
      origin.position.set(-0.32, -0.25, 0.005);
      scene.add(origin);
      const robot = new THREE.Group();
      scene.add(robot);
      const bodyGroups = model.names.map((_, i) => {
        const group = new THREE.Group();
        group.userData.body = i;
        robot.add(group);
        return group;
      });
      const material = (color: string, metalness = 0.25) =>
        new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness });
      const bodyMaterials: {
        body: number;
        material: THREE.MeshStandardMaterial;
      }[] = [];
      const addMesh = (
        body: number,
        geometry: THREE.BufferGeometry,
        color: string,
        position?: number[],
      ) => {
        const mat = material(color);
        const mesh = new THREE.Mesh(geometry, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.body = body;
        if (position) mesh.position.fromArray(position);
        bodyGroups[body].add(mesh);
        bodyMaterials.push({ body, material: mat });
        return mesh;
      };
      const builtIn = model.scene === "arm" || model.scene === "dual_arm";
      for (const visual of model.visuals) {
        const linkName = model.names[visual.body];
        if (
          builtIn &&
          (linkName === "head" ||
            linkName === "world" ||
            linkName.endsWith("pedestal"))
        )
          continue;
        let geometry: THREE.BufferGeometry;
        const d = visual.dimensions;
        if (visual.shape === "box")
          geometry = new RoundedBoxGeometry(
            d[0],
            d[1] * (builtIn ? 1.2 : 1),
            d[2] * (builtIn ? 1.2 : 1),
            3,
            Math.min(...d) * 0.13,
          );
        else if (visual.shape === "sphere")
          geometry = new THREE.SphereGeometry(d[0], 32, 20);
        else {
          geometry = new THREE.CylinderGeometry(d[0], d[0], d[1], 32);
          geometry.rotateX(Math.PI / 2);
        }
        const color =
          builtIn && linkName.includes("wrist") ? "#7f919e" : "#dce3e7";
        const mesh = addMesh(visual.body, geometry, color, visual.origin);
        mesh.setRotationFromEuler(new THREE.Euler(...visual.rpy, "ZYX"));
      }
      for (const joint of model.joints) {
        if (model.scene === "drop") continue;
        const size = builtIn
          ? joint.name.includes("wrist") || joint.name.includes("tool")
            ? 0.046
            : 0.06
          : 0.035;
        const axis = new THREE.Vector3(
          ...(model.axes?.[joint.body] ?? [0, 0, 1]),
        ).normalize();
        const hub = addMesh(
          joint.body,
          new THREE.CylinderGeometry(size, size, size * 1.6, 40),
          "#22313d",
        );
        hub.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
        const cap = addMesh(
          joint.body,
          new THREE.CylinderGeometry(size * 0.82, size * 0.82, 0.008, 40),
          "#9caebc",
        );
        cap.quaternion.copy(hub.quaternion);
        cap.position.copy(axis.clone().multiplyScalar(size * 0.82));
        const center = addMesh(
          joint.body,
          new THREE.CylinderGeometry(size * 0.27, size * 0.27, 0.01, 24),
          "#263b49",
        );
        center.quaternion.copy(hub.quaternion);
        center.position.copy(axis.clone().multiplyScalar(size * 0.91));
      }
      addResearchRobotDetails(model, addMesh);
      const updateManipulation = createManipulationVisuals(
        scene,
        bodyGroups,
        model,
      );
      for (const sphere of model.spheres)
        addMesh(
          sphere.body,
          new THREE.SphereGeometry(sphere.radius, 40, 24),
          "#e77c50",
          sphere.center,
        );
      // Imported models may intentionally have no visual geometry.
      if (!model.visuals.length && !model.spheres.length) {
        model.names.forEach((_, i) =>
          addMesh(i, new THREE.SphereGeometry(0.018, 16, 12), "#81988a"),
        );
      }
      const toolAxes = new THREE.AxesHelper(0.11);
      bodyGroups[model.arms[0]?.body ?? bodyGroups.length - 1]?.add(toolAxes);
      const target = new THREE.Group();
      scene.add(target);
      const targetBall = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 24, 16),
        new THREE.MeshBasicMaterial({
          color: "#da683c",
          wireframe: true,
          transparent: true,
          opacity: 0.6,
        }),
      );
      target.add(targetBall);
      for (let i = 0; i < 3; i++) {
        const points = [new THREE.Vector3(), new THREE.Vector3()];
        points[0].setComponent(i, -0.05);
        points[1].setComponent(i, 0.05);
        target.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: "#bc512d" }),
          ),
        );
      }
      const toolTargets = model.arms.map((arm, i) => {
        const marker = i === 0 ? target : target.clone(true);
        if (i > 0) scene.add(marker);
        return { arm, marker };
      });
      const trailArray = new Float32Array(600 * 3);
      const trailGeometry = new THREE.BufferGeometry();
      trailGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(trailArray, 3),
      );
      trailGeometry.setDrawRange(0, 0);
      const trail = new THREE.Line(
        trailGeometry,
        new THREE.LineBasicMaterial({
          color: "#738e7d",
          transparent: true,
          opacity: 0.65,
        }),
      );
      scene.add(trail);
      const ray = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      let pointerStart = [0, 0];
      const down = (event: PointerEvent) => {
        pointerStart = [event.clientX, event.clientY];
      };
      const up = (event: PointerEvent) => {
        if (
          Math.hypot(
            event.clientX - pointerStart[0],
            event.clientY - pointerStart[1],
          ) > 4
        )
          return;
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.set(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          (-(event.clientY - rect.top) / rect.height) * 2 + 1,
        );
        ray.setFromCamera(pointer, camera);
        const hit = ray
          .intersectObject(robot, true)
          .find((hit) => typeof hit.object.userData.body === "number");
        if (hit) current.current.onSelect(hit.object.userData.body as number);
      };
      renderer.domElement.addEventListener("pointerdown", down);
      renderer.domElement.addEventListener("pointerup", up);
      // One WebGL renderer is reused across all sensors to avoid exhausting contexts.
      const sensor = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
      });
      sensor.toneMapping = THREE.ACESFilmicToneMapping;
      sensor.toneMappingExposure = 1.05;
      const configs = current.current.snapshot.episode?.cameras ?? [];
      const sensors = configs.map((config) => {
        const wrapper = document.createElement("div");
        const label = document.createElement("div");
        label.textContent = `${config.name} · ${config.mount ?? "world"}`;
        label.className = "sensor-heading";
        const preview = document.createElement("canvas");
        preview.width = config.width;
        preview.height = config.height;
        preview.setAttribute("role", "img");
        preview.setAttribute("aria-label", `${config.name} RGB observation`);
        wrapper.append(label, preview);
        sensorMount.current?.appendChild(wrapper);
        const camera = new THREE.PerspectiveCamera(
          config.vertical_fov_degrees,
          config.width / config.height,
          config.near,
          config.far,
        );
        const guide = new THREE.Group();
        const depth = 0.16,
          height =
            Math.tan((config.vertical_fov_degrees * Math.PI) / 360) * depth;
        const width = (height * config.width) / config.height;
        const corners = [
          new THREE.Vector3(-width, -height, -depth),
          new THREE.Vector3(width, -height, -depth),
          new THREE.Vector3(width, height, -depth),
          new THREE.Vector3(-width, height, -depth),
        ];
        const lines: THREE.Vector3[] = [];
        corners.forEach((point, i) => {
          lines.push(new THREE.Vector3(), point, point, corners[(i + 1) % 4]);
        });
        guide.add(
          new THREE.LineSegments(
            new THREE.BufferGeometry().setFromPoints(lines),
            new THREE.LineBasicMaterial({
              color: config.name.startsWith("head") ? "#a15b27" : "#3b7891",
            }),
          ),
        );
        const housing = new THREE.Mesh(
          new THREE.BoxGeometry(0.04, 0.025, 0.025),
          new THREE.MeshBasicMaterial({ color: "#365161" }),
        );
        housing.position.z = 0.0125;
        guide.add(housing);
        scene.add(guide);
        return { config, camera, preview, wrapper, guide };
      });
      const applyPose = (snapshot: Snapshot) => {
        updateManipulation(snapshot);
        const matrix = new THREE.Matrix4();
        bodyGroups.forEach((group, i) => {
          const r = snapshot.state.rotations[i],
            p = snapshot.state.positions[i];
          if (!r || !p) return;
          group.position.fromArray(p);
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
          group.quaternion.setFromRotationMatrix(matrix);
        });
        toolTargets.forEach(({ arm, marker }) => {
          const target = snapshot.state.targets[arm.name];
          marker.visible = model.scene !== "drop" && !!target;
          if (target) marker.position.fromArray(target);
        });
      };
      const renderSensor = (snapshot: Snapshot, capture = false) => {
        applyPose(snapshot);
        const visibility = [
          grid.visible,
          origin.visible,
          toolAxes.visible,
          trail.visible,
        ];
        grid.visible = false;
        origin.visible = false;
        toolAxes.visible = false;
        trail.visible = false;
        bodyMaterials.forEach(
          (entry) => (entry.material.emissiveIntensity = 0),
        );
        const images: Record<string, string> = {};
        const ground_truth_depth: Record<string, string> = {};
        sensors.forEach(({ guide }) => (guide.visible = false));
        for (const {
          config,
          camera: sensorCamera,
          preview,
          guide,
        } of sensors) {
          const position = new THREE.Vector3(...config.position);
          const lookAt = new THREE.Vector3(...config.look_at);
          const up = new THREE.Vector3(...config.up);
          if (config.body != null) {
            const r = snapshot.state.rotations[config.body];
            const translation = new THREE.Vector3(
              ...snapshot.state.positions[config.body],
            );
            const rotation = new THREE.Matrix3().set(
              ...(r.flat() as [
                number,
                number,
                number,
                number,
                number,
                number,
                number,
                number,
                number,
              ]),
            );
            position.applyMatrix3(rotation).add(translation);
            lookAt.applyMatrix3(rotation).add(translation);
            up.applyMatrix3(rotation);
          }
          sensorCamera.position.copy(position);
          sensorCamera.up.copy(up);
          sensorCamera.lookAt(lookAt);
          guide.position.copy(position);
          guide.quaternion.copy(sensorCamera.quaternion);
          sensor.setSize(config.width, config.height, false);
          sensor.render(scene, sensorCamera);
          preview.getContext("2d")?.drawImage(sensor.domElement, 0, 0);
          if (capture) {
            images[config.name] = sensor.domElement.toDataURL("image/png");
            ground_truth_depth[config.name] = captureDepth(
              sensor,
              scene,
              sensorCamera,
              config.width,
              config.height,
            );
          }
        }
        [grid.visible, origin.visible, toolAxes.visible, trail.visible] =
          visibility;
        sensors.forEach(({ guide }) => (guide.visible = true));
        return { images, ground_truth_depth };
      };
      captureFrame.current = (snapshot) => {
        return renderSensor(snapshot, true);
      };
      const resize = new ResizeObserver(() => {
        const width = host.clientWidth,
          height = host.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / Math.max(height, 1);
        camera.updateProjectionMatrix();
      });
      resize.observe(host);
      let animation = 0;
      let sensorTime = 0;
      const matrix = new THREE.Matrix4();
      const render = () => {
        const {
          snapshot,
          selected,
          showAxes,
          showGrid,
          trail: points,
        } = current.current;
        updateManipulation(snapshot);
        const state = snapshot.state;
        bodyGroups.forEach((group, i) => {
          const r = state.rotations[i],
            p = state.positions[i];
          if (!r || !p) return;
          group.position.fromArray(p);
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
          group.quaternion.setFromRotationMatrix(matrix);
        });
        bodyMaterials.forEach((entry) => {
          entry.material.emissive.set(
            entry.body === selected ? "#815025" : "#000000",
          );
          entry.material.emissiveIntensity = 0.18;
        });
        toolTargets.forEach(({ arm, marker }) => {
          // An embodiment update can arrive before this scene's effect is replaced.
          const target = state.targets[arm.name];
          marker.visible = model.scene !== "drop" && !!target;
          if (target) marker.position.fromArray(target);
        });
        grid.visible = showGrid;
        origin.visible = showAxes;
        toolAxes.visible = showAxes;
        const visible = points.slice(-600);
        visible.forEach((point, i) => trailArray.set(point, i * 3));
        trailGeometry.attributes.position.needsUpdate = true;
        trailGeometry.setDrawRange(0, visible.length);
        trailGeometry.computeBoundingSphere();
        if (performance.now() - sensorTime > 150) {
          renderSensor(snapshot);
          sensorTime = performance.now();
        }
        bodyMaterials.forEach(
          (entry) => (entry.material.emissiveIntensity = 0.18),
        );
        controls.update();
        renderer.render(scene, camera);
        animation = requestAnimationFrame(render);
      };
      render();
      return () => {
        cancelAnimationFrame(animation);
        resize.disconnect();
        controls.dispose();
        renderer.domElement.removeEventListener("pointerdown", down);
        renderer.domElement.removeEventListener("pointerup", up);
        scene.traverse((object) => {
          const mesh = object as THREE.Mesh;
          mesh.geometry?.dispose();
          if (mesh.material)
            (Array.isArray(mesh.material)
              ? mesh.material
              : [mesh.material]
            ).forEach((m) => m.dispose());
        });
        renderer.dispose();
        renderer.domElement.remove();
        sensor.dispose();
        sensors.forEach(({ wrapper }) => wrapper.remove());
        captureFrame.current = () => {
          throw new Error("Camera is still initializing");
        };
      };
    }, [props.snapshot.epoch]);
    return (
      <>
        <div
          className={`viewport-canvas ${previews ? "with-sensors" : ""}`}
          ref={mount}
        >
          {failure && <div className="webgl-error">{failure}</div>}
        </div>
        <div
          className={`sensor-card sensor-strip ${previews ? "" : "collapsed"}`}
        >
          <div className="sensor-heading">
            <span className="sensor-led" />
            CAMERA RIG{" "}
            <span>
              {props.snapshot.episode?.cameras.length ?? 0} synchronized views
            </span>
            <button
              className="sensor-toggle"
              aria-label={
                previews ? "Hide camera previews" : "Show camera previews"
              }
              onClick={() => setPreviews(!previews)}
            >
              {previews ? "Hide" : "Show"}
            </button>
          </div>
          <div
            className="sensor-canvas"
            ref={sensorMount}
            style={{
              display: previews ? "grid" : "none",
              gridTemplateColumns: `repeat(${Math.min(props.snapshot.episode?.cameras.length || 1, 4)}, minmax(0,1fr))`,
            }}
          />
          <div className="sensor-foot" hidden={!previews}>
            All views captured at one simulation timestamp
          </div>
        </div>
      </>
    );
  },
);
