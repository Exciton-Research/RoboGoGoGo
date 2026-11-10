import { useWorkspace } from "../state/WorkspaceProvider";

export function PoseDraftControls() {
  const {
    goalDirty,
    posePreview,
    posePreviewError,
    command,
    disabled,
    state,
    setGoalDirty,
    setGoals,
  } = useWorkspace();
  if (!goalDirty)
    return (
      <p className="control-note">
        Adjust an arm or torso parameter to preview its pose. The outline shows
        your draft; Apply pose moves the robot through a checked path.
      </p>
    );
  const active =
    !!state?.motion &&
    ["ready", "running", "paused"].includes(state.motion.status);
  return (
    <section className="pose-draft" aria-label="Draft robot pose">
      <strong>Pose preview · not executed</strong>
      <p role="status">
        {posePreviewError ||
          (!posePreview
            ? "Checking draft pose…"
            : posePreview.valid
              ? "Endpoint clear. Apply will check the complete motion path."
              : `Unsafe endpoint: ${posePreview.pair.join(" / ")}. Adjust the red outline before applying.`)}
      </p>
      {state?.running && (
        <button
          className="button"
          disabled={disabled}
          onClick={() => void command("pause")}
        >
          Pause to edit pose
        </button>
      )}
      {active && (
        <button
          className="button"
          disabled={disabled}
          onClick={() => void command("cancel_plan")}
        >
          Cancel current motion
        </button>
      )}
      <div className="pose-draft-actions">
        <button
          className="button"
          disabled={
            disabled || !!state?.running || active || !posePreview?.valid
          }
          onClick={() =>
            void command("joints", { values: posePreview?.values, start: true })
          }
        >
          Apply pose
        </button>
        <button
          className="button"
          disabled={disabled}
          onClick={() => {
            setGoalDirty(false);
            setGoals(state?.goal ?? []);
          }}
        >
          Discard edits
        </button>
      </div>
    </section>
  );
}
