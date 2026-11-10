import { useEffect, useState } from "react";
import { Crosshair, Move3D } from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";
import type { Vec3 } from "../types";

export function WholeBodyControls() {
  const { model, state, snapshot, command, disabled } = useWorkspace();
  const [target, setTarget] = useState<Vec3>([0.7, 0, 1.1]);
  const handCenter = (): Vec3 =>
    [0, 1, 2].map(
      (axis) =>
        (model?.arms.reduce(
          (sum, arm) => sum + (state?.positions[arm.body]?.[axis] ?? 0),
          0,
        ) ?? 0) / (model?.arms.length || 1),
    ) as Vec3;
  useEffect(() => {
    if (!model || !state) return;
    if (state.motion?.kind === "whole_body" && state.motion.target_center) {
      setTarget(state.motion.target_center);
      return;
    }
    const center = handCenter();
    setTarget([center[0] + 0.35, center[1], center[2] - 0.12]);
  }, [snapshot?.epoch]);
  const motion = state?.motion;
  const active =
    !!motion && ["ready", "running", "paused"].includes(motion.status);
  return (
    <section className="whole-body-card" aria-label="Whole-body reach controls">
      <div className="whole-body-heading">
        <Move3D size={18} />
        <div>
          <span className="section-kicker">COORDINATED REACH</span>
          <h3>One target. The whole robot.</h3>
        </div>
      </div>
      <p>
        Wheels position the base while the torso and both arms reach together.
        Target is the midpoint between the hands, in world coordinates.
      </p>
      <div className="whole-body-fields">
        {(["X", "Y", "Z"] as const).map((axis, i) => (
          <label key={axis}>
            {axis}
            <input
              aria-label={`Whole-body target ${axis}`}
              type="number"
              step="0.02"
              value={Number(target[i].toFixed(3))}
              disabled={disabled || active}
              onChange={(event) =>
                setTarget(
                  (old) =>
                    old.map((value, j) =>
                      j === i ? Number(event.target.value) : value,
                    ) as Vec3,
                )
              }
            />
            <small>m</small>
          </label>
        ))}
      </div>
      <button
        className="button whole-body-start"
        disabled={disabled || active || !target.every(Number.isFinite)}
        onClick={() =>
          void command("whole_body", { values: target, start: true })
        }
      >
        <Crosshair size={15} />
        Reach with whole body
      </button>
      <button
        className="button whole-body-secondary"
        disabled={disabled || active}
        onClick={() => setTarget(handCenter())}
      >
        Use current hands
      </button>
      {motion?.kind === "whole_body" && active && (
        <div className="whole-body-actions">
          <button
            className="button"
            disabled={disabled}
            onClick={() => void command(state?.running ? "pause" : "run")}
          >
            {state?.running ? "Pause reach" : "Resume reach"}
          </button>
          <button
            className="button"
            disabled={disabled}
            onClick={() => void command("cancel_plan")}
          >
            Cancel reach
          </button>
        </div>
      )}
      <div className="whole-body-tags">
        <span>4 wheel drives</span>
        <span>Articulated torso</span>
        <span>2 × 7-DoF arms</span>
      </div>
      {motion?.kind === "whole_body" && (
        <p role="status">
          <strong>{motion.status}</strong> · segment {motion.waypoint}/
          {motion.waypoint_count}
          {motion.base_error_m !== undefined &&
            ` · base error ${(motion.base_error_m * 1000).toFixed(0)} mm`}
        </p>
      )}
      <small>
        Local position-only reach. Preserves final hand spacing; no rigid-object
        constraint. Self-collision and support guards stay active. No
        environment obstacle or payload planning.
      </small>
    </section>
  );
}
