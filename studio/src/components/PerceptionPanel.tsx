import { useEffect, useState } from "react";
import type { EpisodeSummary } from "../types";

export interface StereoReport {
  episode_id: string;
  observation: number;
  time: number;
  left: string;
  right: string;
  baseline_m: number;
  valid_fraction: number;
  depth_scale_m: number;
  estimated: string;
  ground_truth: string | null;
  disparity: string;
  metrics: {
    coverage: number;
    mae_m: number | null;
    rmse_m: number | null;
  } | null;
}
export function PerceptionPanel({
  episode,
  disabled,
  analyze,
  report,
}: {
  episode: EpisodeSummary | null;
  disabled: boolean;
  analyze: (left: string, right: string) => void;
  report: StereoReport | null;
}) {
  const [left, setLeft] = useState("head_left"),
    [right, setRight] = useState("head_right");
  useEffect(() => {
    setLeft(episode?.cameras[0]?.name ?? "");
    setRight(episode?.cameras[1]?.name ?? "");
  }, [episode?.id]);
  const current = report?.episode_id === episode?.id ? report : null;
  return (
    <div className="vla-block">
      <div className="section-heading">
        <span>STEREO PERCEPTION</span>
        <span className="baseline-tag">CLASSICAL</span>
      </div>
      <p className="vla-caption">
        Two calibrated views estimate distance. Ground truth is reserved for
        evaluation.
      </p>
      <div className="stereo-selectors">
        {(["Left", "Right"] as const).map((label, i) => (
          <label key={label}>
            {label}
            <select
              aria-label={`${label} stereo camera`}
              value={i ? right : left}
              onChange={(e) => (i ? setRight : setLeft)(e.target.value)}
              disabled={disabled}
            >
              {episode?.cameras.map((c) => (
                <option key={c.name}>{c.name}</option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <button
        className="button capture-button"
        disabled={disabled || !left || !right || left === right}
        onClick={() => analyze(left, right)}
      >
        Capture & evaluate stereo
      </button>
      {current && (
        <div className="stereo-results">
          <p className="vla-caption">
            Frame {current.observation} · {current.time.toFixed(2)} s · baseline{" "}
            {(current.baseline_m * 1000).toFixed(1)} mm
          </p>
          <figure>
            <img
              src={current.estimated}
              alt="Estimated stereo depth with invalid pixels dark"
            />
            <figcaption>
              Estimated depth · {(current.valid_fraction * 100).toFixed(1)}%
              valid
            </figcaption>
          </figure>
          {current.ground_truth && (
            <figure>
              <img
                src={current.ground_truth}
                alt="Renderer ground-truth depth"
              />
              <figcaption>Renderer ground truth · privileged</figcaption>
            </figure>
          )}
          <details>
            <summary>Disparity</summary>
            <img src={current.disparity} alt="Stereo disparity map" />
          </details>
          <p className="vla-caption">
            Depth colors: near blue → far red, 0–
            {current.depth_scale_m.toFixed(2)} m. Dark = unknown. Both maps use
            the rectified left camera.
          </p>
          {current.metrics && (
            <p className="vla-caption">
              MAE{" "}
              {current.metrics.mae_m == null
                ? "unavailable"
                : `${(current.metrics.mae_m * 1000).toFixed(1)} mm`}{" "}
              · RMSE{" "}
              {current.metrics.rmse_m == null
                ? "unavailable"
                : `${(current.metrics.rmse_m * 1000).toFixed(1)} mm`}
              <br />
              Ground-truth coverage{" "}
              {(current.metrics.coverage * 100).toFixed(1)}%
            </p>
          )}
          <p className="vla-caption">
            Textureless surfaces, occlusions and reflections can leave gaps or
            incorrect matches. This is a classical baseline.
          </p>
        </div>
      )}
    </div>
  );
}
