import type { CameraConfig, RobotModel, Vec3 } from "../types";

interface Props {
  cameras: CameraConfig[];
  model: RobotModel | null;
  disabled: boolean;
  onChange: (cameras: CameraConfig[]) => void;
}

function angles(camera: CameraConfig): Vec3 {
  if (camera.rpy) return camera.rpy;
  const f = camera.look_at.map((v, i) => v - camera.position[i]);
  const length = Math.hypot(...f);
  const [x, y, z] = f.map((v) => v / length);
  const [ux, uy, uz] = camera.up;
  const left = [uy * z - uz * y, uz * x - ux * z, ux * y - uy * x];
  const l = Math.hypot(...left);
  const [lx, ly, lz] = left.map((v) => v / l);
  const upZ = x * ly - y * lx;
  return [
    Math.atan2(lz, upZ),
    Math.asin(Math.max(-1, Math.min(1, -z))),
    Math.atan2(y, x),
  ];
}

export function CameraMountEditor({
  cameras,
  model,
  disabled,
  onChange,
}: Props) {
  const update = (index: number, patch: Partial<CameraConfig>) =>
    onChange(
      cameras.map((camera, i) =>
        i === index ? { ...camera, rpy: angles(camera), ...patch } : camera,
      ),
    );
  return (
    <div className="camera-mount-editor">
      <div className="task-variant">
        <label htmlFor="camera-preset">Mounting preset</label>
        <select
          id="camera-preset"
          aria-label="Camera mounting preset"
          disabled={disabled}
          value="custom"
          onChange={(e) => {
            const preset = model?.camera_presets[e.target.value];
            if (preset) onChange(preset);
          }}
        >
          <option value="custom">Current / custom mounts</option>
          {model?.camera_presets.head_wrists && (
            <option value="head_wrists">Head stereo + wrist cameras</option>
          )}
          {model?.camera_presets.head && (
            <option value="head">Head stereo pair</option>
          )}
        </select>
      </div>
      <p className="vla-caption">
        Demo dimensions, not hardware calibration. Edit measured mounts below,
        then start a new episode. The demo head is a fixed mast.
      </p>
      {cameras.map((camera, index) => (
        <details key={camera.name} className="camera-mount">
          <summary>
            {camera.name} <small>on {camera.mount ?? "world"}</small>
          </summary>
          <label>
            Parent link
            <select
              aria-label={`${camera.name} parent link`}
              value={camera.mount ?? ""}
              disabled={disabled}
              onChange={(e) => update(index, { mount: e.target.value || null })}
            >
              <option value="">World / external fixture</option>
              {model?.names.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <span className="mount-caption">
            Offset from parent link (meters)
          </span>
          <div className="mount-fields">
            {["X", "Y", "Z"].map((axis, i) => (
              <label key={axis}>
                {axis}
                <input
                  type="number"
                  step="0.005"
                  aria-label={`${camera.name} offset ${axis}`}
                  value={camera.position[i]}
                  disabled={disabled}
                  onChange={(e) => {
                    const position = [...camera.position] as Vec3;
                    position[i] = Number(e.target.value);
                    update(index, { position });
                  }}
                />
              </label>
            ))}
          </div>
          <span className="mount-caption">Mount orientation (degrees)</span>
          <div className="mount-fields">
            {["Roll", "Pitch", "Yaw"].map((axis, i) => (
              <label key={axis}>
                {axis}
                <input
                  type="number"
                  step="1"
                  aria-label={`${camera.name} ${axis}`}
                  value={Number(
                    ((angles(camera)[i] * 180) / Math.PI).toFixed(2),
                  )}
                  disabled={disabled}
                  onChange={(e) => {
                    const rpy = [...angles(camera)] as Vec3;
                    rpy[i] = (Number(e.target.value) * Math.PI) / 180;
                    update(index, { rpy });
                  }}
                />
              </label>
            ))}
          </div>
          <label>
            Vertical FOV (degrees)
            <input
              type="number"
              min="2"
              max="169"
              aria-label={`${camera.name} field of view`}
              value={camera.vertical_fov_degrees}
              disabled={disabled}
              onChange={(e) =>
                update(index, { vertical_fov_degrees: Number(e.target.value) })
              }
            />
          </label>
        </details>
      ))}
      <p className="vla-caption">
        +X forward · +Y left · +Z up. Positive pitch looks down. Camera outlines
        show the active episode mounts; edits apply on restart.
      </p>
    </div>
  );
}
