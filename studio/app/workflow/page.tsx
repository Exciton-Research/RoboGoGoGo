"use client";

import { useEffect, useState } from "react";
import { request } from "../../src/api";
import styles from "../workflow.module.css";

type Workflow = {
  robot: {
    name: string;
    model: string | null;
    missing: string[];
    camera_roles: string[];
  };
  hardware: { connected: boolean; calibrated: boolean; status: string };
  physical_execution_enabled: boolean;
  stages: { id: string; title: string; state: string; detail: string }[];
  episodes: {
    id: string;
    instruction: string;
    actions: number;
    environment?: string;
  }[];
};
type Validation = {
  transitions: number;
  environment: string;
  cameras: string[];
  sha256: string;
};
const links: Record<string, { href: string; label: string }> = {
  robot: { href: "/hardware/", label: "Inspect hardware" },
  simulation: { href: "/simulation/", label: "Open simulation" },
  data: { href: "#episodes", label: "Review episodes" },
  runtime: { href: "/execution/", label: "Test command execution" },
};

export default function WorkflowPage() {
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, Validation | string>>(
    {},
  );
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const value = await request<Workflow>("/workflow");
        if (active) {
          setWorkflow(value);
          setError("");
        }
      } catch (failure) {
        if (active)
          setError(
            failure instanceof Error ? failure.message : "Workflow unavailable",
          );
      }
    };
    void load();
    const timer = setInterval(() => void load(), 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  async function validate(id: string) {
    setBusy(id);
    try {
      const report = await request<Validation>(`/data/${id}/validate`, {});
      setResults((previous) => ({ ...previous, [id]: report }));
    } catch (failure) {
      setResults((previous) => ({
        ...previous,
        [id]: failure instanceof Error ? failure.message : "Validation failed",
      }));
    } finally {
      setBusy(null);
    }
  }
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a className={styles.brand} href="/">
          robogogogo<span>.</span>
        </a>
        <nav aria-label="Workspace">
          <a href="/simulation/">Simulation</a>
          <a href="/hardware/">Hardware</a>
          <a href="/execution/">Command test</a>
        </nav>
      </header>
      <section className={styles.intro}>
        <p className={styles.eyebrow}>DUAL-ARM DEVELOPMENT</p>
        <h1>
          One robot.
          <br />A complete workflow.
        </h1>
        <p>
          Connect your robot, collect demonstrations, train a policy and bring
          it back to the hardware. Track what is available and what still needs
          integration.
        </p>
      </section>
      {error && (
        <p className={styles.error} role="alert">
          Connection issue: {error}. Displayed status may be out of date.
        </p>
      )}
      {!workflow && !error && <p role="status">Loading workspace…</p>}
      {workflow && (
        <>
          <section className={styles.robot} aria-label="Deployment robot">
            <div>
              <p className={styles.eyebrow}>YOUR ROBOT</p>
              <h2>{workflow.robot.name}</h2>
              <p>{workflow.robot.model ?? "Physical model not selected"}</p>
            </div>
            <div>
              <strong>Two arms · two grippers</strong>
              <p>Head stereo + left and right wrist cameras</p>
              <span className={styles.badge}>
                {workflow.hardware.connected
                  ? "Receiving selected robot telemetry"
                  : "Hardware not connected"}
              </span>
            </div>
            <div>
              <strong>Physical execution</strong>
              <p>
                {workflow.physical_execution_enabled
                  ? "Enabled"
                  : "Unavailable — read-only integration"}
              </p>
              <a href="/hardware/">Connection & calibration →</a>
            </div>
          </section>
          {workflow.robot.missing.length > 0 && (
            <p className={styles.pending}>
              Robot setup pending:{" "}
              {workflow.robot.missing
                .map((name) => name.replaceAll("_", " "))
                .join(", ")}
              . The simulation uses a reference robot until your hardware is
              selected and modeled.
            </p>
          )}
          <section className={styles.grid} aria-label="Development workflow">
            {workflow.stages.map((stage, index) => (
              <article className={styles.card} key={stage.id}>
                <div className={styles.cardTop}>
                  <span className={styles.number}>
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className={styles.badge} data-state={stage.state}>
                    {stage.state}
                  </span>
                </div>
                <h2>{stage.title}</h2>
                <p>{stage.detail}</p>
                {links[stage.id] && (
                  <a href={links[stage.id].href}>{links[stage.id].label} →</a>
                )}
              </article>
            ))}
          </section>
          <section id="episodes" className={styles.episodes}>
            <div className={styles.sectionTitle}>
              <div>
                <p className={styles.eyebrow}>DATA → LEARNING</p>
                <h2>Episode library</h2>
              </div>
              <a href="/simulation/">Collect in simulation →</a>
            </div>
            <p>
              Validate observation/action alignment before using an archive for
              training. Validation does not establish policy quality or hardware
              compatibility.
            </p>
            {workflow.episodes.length === 0 && (
              <div className={styles.empty}>
                No saved episodes yet. Capture observations and actions in
                simulation, then save the episode.
              </div>
            )}
            {workflow.episodes.map((episode) => {
              const result = results[episode.id];
              return (
                <article className={styles.episode} key={episode.id}>
                  <div>
                    <h3>{episode.instruction}</h3>
                    <p>
                      {episode.actions} actions ·{" "}
                      {episode.environment ?? "Origin unknown"}
                    </p>
                  </div>
                  <div className={styles.actions}>
                    <a href={`/api/episodes/episodes/${episode.id}`}>
                      Download
                    </a>
                    <button
                      disabled={busy !== null}
                      onClick={() => void validate(episode.id)}
                    >
                      {busy === episode.id ? "Validating…" : "Validate dataset"}
                    </button>
                  </div>
                  {result && (
                    <p className={styles.result} role="status">
                      {typeof result === "string"
                        ? `Validation failed: ${result}`
                        : `${result.transitions} aligned transitions · ${result.cameras.length} camera streams · ${result.environment} data · SHA-256 ${result.sha256}`}
                    </p>
                  )}
                </article>
              );
            })}
          </section>
        </>
      )}
    </main>
  );
}
