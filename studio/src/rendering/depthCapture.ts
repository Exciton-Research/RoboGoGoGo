import * as THREE from "three";

/** Renderer ground truth: optical Z meters, top-left origin, little-endian float32. */
export function captureDepth(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number,
): string {
  const target = new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
  });
  const material = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
  });
  material.blending = THREE.NoBlending;
  const previous = {
    target: renderer.getRenderTarget(),
    material: scene.overrideMaterial,
    background: scene.background,
    color: renderer.getClearColor(new THREE.Color()),
    alpha: renderer.getClearAlpha(),
  };
  const pixels = new Uint8Array(width * height * 4);
  try {
    scene.overrideMaterial = material;
    scene.background = null;
    renderer.setClearColor(0xffffff, 1);
    renderer.setRenderTarget(target);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
  } finally {
    renderer.setRenderTarget(previous.target);
    renderer.setClearColor(previous.color, previous.alpha);
    scene.overrideMaterial = previous.material;
    scene.background = previous.background;
    target.dispose();
    material.dispose();
  }
  const bytes = new Uint8Array(width * height * 4);
  const output = new DataView(bytes.buffer);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = ((height - 1 - y) * width + x) * 4;
      // Three r181 unpackRGBAToDepth, accounting for normalized byte channels.
      const d =
        pixels[i] / 256 +
        pixels[i + 1] / 65536 +
        pixels[i + 2] / 16777216 +
        pixels[i + 3] / (255 * 16777216);
      const z =
        d >= 1
          ? NaN
          : (camera.near * camera.far) /
            (camera.far - d * (camera.far - camera.near));
      output.setFloat32((y * width + x) * 4, z, true);
    }
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192)
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}
