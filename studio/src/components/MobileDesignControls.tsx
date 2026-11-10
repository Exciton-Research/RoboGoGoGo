import { useWorkspace } from "../state/WorkspaceProvider";

export function MobileDesignControls() {
  const { model, state, goals, setGoals, setGoalDirty, command, disabled } =
    useWorkspace();
  if (model?.scene !== "mobile_dual_arm") return null;
  const bodyNames = [
    "base_x",
    "base_y",
    "base_link",
    "torso_yaw",
    "torso_lower",
    "torso_upper",
    "torso",
  ];
  return (
    <section
      aria-label="Mobile robot design controls"
      style={{ padding: "14px 0", borderBottom: "1px solid var(--line)" }}
    >
      <div className="section-kicker">ARTICULATED MOBILE CONCEPT</div>
      <p className="control-note">
        Four steerable wheels, folding torso, two 7-joint arms. Ideal planar
        support: no tire traction, tipping or validated collision avoidance.
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
      <details>
        <summary style={{ margin: "14px 0", cursor: "pointer" }}>
          Base and torso coordinates
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
          onClick={() => void command("joints", { values: goals })}
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
