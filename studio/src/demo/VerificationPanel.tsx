"use client";

import type { TaskState } from "./types";
import styles from "./demo.module.css";

const stages: Record<string, string> = {
  pickup: "Finding a stable grasp",
  transport: "Lift verified · transporting",
  verifying_lift: "Checking the named object in hand",
  verifying_placement: "Checking release and placement",
  recovering: "Returning and releasing before retry",
  relocalizing: "Inspecting again for a fresh pickup",
  verified: "Placement verified by cameras",
  stopped: "Automatic recovery stopped",
};

export default function VerificationPanel({ state }: { state: TaskState }) {
  const check = state.verification;
  if (!check) return null;
  const failure = check.events
    .slice()
    .reverse()
    .find(
      (event) => event.event === "attempt_failed" || event.event === "stopped",
    );
  return (
    <section
      className={styles.verificationPanel}
      aria-label="Camera verification and recovery"
    >
      <strong>{stages[check.stage] ?? check.stage}</strong>
      {state.replay && (
        <p>Saved episode summary · history covers the full episode.</p>
      )}
      <p>
        Attempt {check.attempt} of {check.max_attempts} · lift{" "}
        {check.lift_verified ? "verified" : "pending"} · placement{" "}
        {check.placement_verified ? "verified" : "pending"}
      </p>
      {failure?.reason && (
        <p>
          {check.placement_verified ? "Recovered after: " : "Last issue: "}
          {failure.reason}
        </p>
      )}
      {check.placement_verified && check.evaluation_success === false && (
        <p role="status">
          The camera confirmed placement, but the independent simulator settling
          check disagreed. This result needs inspection.
        </p>
      )}
      {check.stage === "stopped" && (
        <p>
          Task paused. The recording is saved when it contains steps. Manual
          control remains available without resetting the table.
        </p>
      )}
      <details>
        <summary>Camera checks and recovery history</summary>
        {!state.replay && check.image_revision > 0 && (
          <img
            src={`/api/demo/verification/image?episode_id=${state.episode_id}&revision=${check.image_revision}`}
            alt="Latest camera observation used for target verification"
          />
        )}
        <ol>
          {check.events.map((event, index) => (
            <li key={index}>
              Attempt {event.attempt} · step {event.step}:{" "}
              {event.event.replaceAll("_", " ")}
              {event.views ? ` (${event.views.join(" + ")})` : ""}
              {event.score !== undefined
                ? ` · score ${event.score.toFixed(2)}`
                : ""}
            </li>
          ))}
        </ol>
      </details>
    </section>
  );
}
