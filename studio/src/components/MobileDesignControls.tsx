import { useState } from "react";
import { useWorkspace } from "../state/WorkspaceProvider";

export function MobileDesignControls() {
  const { model, state, goals, setGoals, setGoalDirty, command, disabled } =
    useWorkspace();
  const [friction, setFriction] = useState(0.7);
  if (model?.scene !== "mobile_dual_arm") return null;
  const bodyNames = ["torso_yaw", "torso_lower", "torso_upper", "torso"];
  return (
    <section
      aria-label="Mobile robot design controls"
      style={{ padding: "14px 0", borderBottom: "1px solid var(--line)" }}
    >
      <div className="section-kicker">ARTICULATED MOBILE CONCEPT</div>
      <p className="control-note">
        Coupled wheel drive with tire slip and friction. Flat-ground model with
        a support-polygon guard; no rollover physics or collision avoidance.
      </p>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {(["standing", "low", "reach"] as const).map((posture) => (
          <button
            key={posture}
            className="button"
            disabled={disabled}
            onClick={() => void command("posture", { posture })}
          >
            Set {posture} posture
          </button>
        ))}
        <button
          className="button"
          disabled={disabled}
          onClick={() => void command(state?.running ? "pause" : "run")}
        >
          {state?.running ? "Pause motion" : "Run posture motion"}
        </button>
      </div>
      <div style={{ marginTop: 14 }}>
        <div className="section-kicker">WHEEL DRIVE · BODY FRAME</div>
        <label>
          Tire friction μ{" "}
          <input
            aria-label="Tire friction coefficient"
            type="number"
            min={0}
            max={1.5}
            step={0.05}
            value={friction}
            onChange={(event) => setFriction(Number(event.target.value))}
          />
        </label>
        <div
          style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}
        >
          {(
            [
              ["Forward", [0.2, 0, 0]],
              ["Reverse", [-0.2, 0, 0]],
              ["Strafe left", [0, 0.15, 0]],
              ["Strafe right", [0, -0.15, 0]],
              ["Turn left", [0, 0, 0.2]],
              ["Turn right", [0, 0, -0.2]],
            ] as const
          ).map(([label, values]) => (
            <button
              className="button"
              key={label}
              disabled={disabled}
              onClick={() =>
                void command("drive", { values, duration: 3, friction })
              }
            >
              {label} · 3 s
            </button>
          ))}
          <button
            className="button"
            disabled={disabled}
            onClick={() => void command("brake")}
          >
            Brake wheels
          </button>
        </div>
        <p className="control-note">
          Set a drive goal, then Run. Each goal expires after 3 simulation
          seconds. Brake sets wheel speed to zero; traction determines stopping
          distance. Pause freezes simulation and clears drive.
        </p>
        {state?.mobile && (
          <p role="status" className="control-note">
            Stability: {state.mobile.guard} · margin{" "}
            {(state.mobile.support_margin_m * 1000).toFixed(0)} mm (reserve 35
            mm). Tire loads:{" "}
            {state.mobile.normal_loads_n.map((v) => v.toFixed(0)).join(" / ")}{" "}
            N. Slip:{" "}
            {state.mobile.longitudinal_slip_m_s
              .map((v) => v.toFixed(3))
              .join(" / ")}{" "}
            m/s.
          </p>
        )}
      </div>
      <details>
        <summary style={{ margin: "14px 0", cursor: "pointer" }}>
          Torso joint goals
        </summary>
        {model.joints.map(
          (joint, i) =>
            bodyNames.includes(joint.name) && (
              <label
                key={joint.name}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 8,
                  margin: "8px 0",
                }}
              >
                {joint.name.replaceAll("_", " ")} (
                {joint.kind === "prismatic" ? "m" : "rad"})
                <input
                  aria-label={`${joint.name} design goal`}
                  type="number"
                  style={{ width: 75 }}
                  min={joint.lower ?? -3.14}
                  max={joint.upper ?? 3.14}
                  step="0.05"
                  disabled={disabled}
                  value={Number((goals[i] ?? 0).toFixed(3))}
                  onChange={(event) => {
                    setGoals((old) =>
                      old.map((q, j) =>
                        j === i ? Number(event.target.value) : q,
                      ),
                    );
                    setGoalDirty(true);
                  }}
                />
              </label>
            ),
        )}
        <button
          className="button"
          disabled={disabled}
          onClick={() =>
            void command("joints", {
              values: goals.map((q, i) =>
                bodyNames.includes(model.joints[i].name)
                  ? q
                  : (state?.goal[i] ?? q),
              ),
            })
          }
        >
          Apply body goals
        </button>
      </details>
      <p className="control-note">
        Presets set torque-controller goals; press Run to move. These are design
        exploration controls. The separate command test fixture still has
        six-joint arms.
      </p>
    </section>
  );
}
