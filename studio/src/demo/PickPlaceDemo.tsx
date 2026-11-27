"use client";

import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import { request } from "../api";
import TaskScene from "./TaskScene";
import type { SceneData, TaskState } from "./types";
import styles from "./demo.module.css";

export default function PickPlaceDemo() {
  const [scene, setScene] = useState<SceneData | null>(null);
  const [state, setState] = useState<TaskState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState(0);
  const [variant, setVariant] = useState(0);
  useEffect(() => {
    let active = true,
      timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const scene = await request<SceneData>("/demo/scene");
        if (!active) return;
        setScene(scene);
        setState(scene.state);
        setError("");
        void poll();
      } catch (failure) {
        if (active)
          setError(
            failure instanceof Error ? failure.message : "Demo unavailable",
          );
      }
    }
    async function poll() {
      try {
        const next = await request<TaskState>("/demo/state");
        if (!active) return;
        setState(next);
        setError("");
      } catch {
        if (active) setError("Connection interrupted. Reconnecting…");
      }
      if (active) timer = setTimeout(() => void poll(), 60);
    }
    void load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  async function command(action: string, nextVariant = variant) {
    setBusy(true);
    try {
      const next = await request<TaskState>("/demo/command", {
        command: action,
        variant: nextVariant,
      });
      setState(next);
      setVariant(nextVariant);
      setError("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Command failed");
    } finally {
      setBusy(false);
    }
  }
  const running = state?.status === "running";
  const finished = state?.status === "complete" || state?.status === "failed";
  const disabled = busy || !state || Boolean(error);
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a href="/" className={styles.brand}>
          robogogogo<span>.</span>
        </a>
        <div className={styles.breadcrumb}>
          ROBOT MANIPULATION <ChevronRight size={12} /> DEMO 01
        </div>
        {scene?.legacy_workspace ? (
          <a className={styles.workspaceLink} href="/simulation/">
            Robot workspace <ArrowUpRight size={14} />
          </a>
        ) : (
          <span className={styles.workspaceLink}>
            Scripted control · simulated physics
          </span>
        )}
      </header>
      <div className={styles.workspace}>
        <section className={styles.stage} aria-label="Robot task scene">
          <div className={styles.stageLabel}>
            <span className={styles.liveDot} /> PANDA / GRASP & PLACE{" "}
            <span>MuJoCo dynamics</span>
          </div>
          {scene && state ? (
            <TaskScene sceneData={scene} state={state} view={view} />
          ) : (
            <div className={styles.loading}>
              <span className={styles.spinner} />
              <h2>Preparing the workbench</h2>
              <p>Loading the Panda robot and contact simulation.</p>
            </div>
          )}
          <button
            className={styles.frameButton}
            aria-label="Reset camera view"
            title="Frame the workbench"
            onClick={() => setView(view + 1)}
          >
            <Maximize2 size={17} />
          </button>
          <div className={styles.orbitHint}>
            Drag to orbit <span>·</span> Scroll to zoom
          </div>
          <div className={styles.objectLegend}>
            <span /> 50 mm block <i /> Placement target
          </div>
        </section>
        <aside className={styles.panel}>
          <div className={styles.taskIntro}>
            <p className={styles.eyebrow}>ONE ARM. ONE TASK.</p>
            <h1>
              Grasp &<br />
              place.
            </h1>
            <p>
              Pick up the orange block and place it inside the marked square.
            </p>
          </div>
          <div
            className={styles.status}
            data-state={state?.status ?? "loading"}
            role="status"
          >
            <span className={styles.statusDot} />{" "}
            {state?.status === "complete"
              ? "Placement complete"
              : state?.status === "failed"
                ? "Task stopped"
                : state?.status === "running"
                  ? "Task in progress"
                  : state?.status === "paused"
                    ? "Paused"
                    : state
                      ? "Ready when you are"
                      : "Preparing simulation"}
            <span>{state ? `${state.time.toFixed(1)} s` : "—"}</span>
          </div>
          <div className={styles.controls}>
            <button
              className={styles.runButton}
              disabled={disabled}
              onClick={() =>
                void command(running ? "pause" : finished ? "reset" : "run")
              }
            >
              {running ? (
                <Pause size={16} />
              ) : finished ? (
                <RotateCcw size={16} />
              ) : (
                <Play size={16} fill="currentColor" />
              )}
              {busy
                ? "Applying…"
                : running
                  ? "Pause task"
                  : finished
                    ? "Reset task"
                    : state?.status === "paused"
                      ? "Resume task"
                      : "Run grasp & place"}
            </button>
            <button
              className={styles.resetButton}
              aria-label="Reset task"
              title="Reset task"
              disabled={busy || !state}
              onClick={() => void command("reset")}
            >
              <RotateCcw size={17} />
            </button>
          </div>
          {error && (
            <div className={styles.error} role="alert">
              <strong>Demo unavailable</strong>
              <p>{error}</p>
              {!scene && <code>robogogogo-demo --setup</code>}
            </div>
          )}
          <ol className={styles.steps} aria-label="Task progress">
            {[
              "Approach & align",
              "Establish grasp",
              "Lift & carry",
              "Place & release",
              "Withdraw & verify",
            ].map((label, index) => {
              const phase = state?.phase ?? 0;
              const group =
                phase < 2
                  ? 0
                  : phase === 2
                    ? 1
                    : phase < 5
                      ? 2
                      : phase < 7
                        ? 3
                        : 4;
              const done = state?.status === "complete" || group > index;
              return (
                <li
                  key={label}
                  data-active={
                    group === index && state?.status !== "ready" && !done
                  }
                  data-done={done}
                >
                  <span>
                    {done ? (
                      <Check size={12} />
                    ) : (
                      String(index + 1).padStart(2, "0")
                    )}
                  </span>
                  <div>
                    {label}
                    <small>
                      {group === index && state?.status === "running"
                        ? state.phases[phase]
                        : done
                          ? "Done"
                          : ""}
                    </small>
                  </div>
                </li>
              );
            })}
          </ol>
          <div className={styles.readouts}>
            <div>
              <span>Object lift</span>
              <strong>
                {state ? Math.round(state.lift_mm) : "—"}
                <small>mm</small>
              </strong>
            </div>
            <div>
              <span>Finger opening</span>
              <strong>
                {state ? Math.round(state.finger_gap_mm) : "—"}
                <small>mm</small>
              </strong>
            </div>
            <div>
              <span>Target distance</span>
              <strong>
                {state ? Math.round(state.placement_error_mm) : "—"}
                <small>mm</small>
              </strong>
            </div>
          </div>
          <p className={styles.outcome}>
            {finished
              ? state.message
              : "The fingers hold the block through contact and friction. Opening them releases it under gravity."}
          </p>
          <button
            className={styles.newPlacement}
            disabled={disabled || running}
            onClick={() => void command("reset", variant + 1)}
          >
            Try a new starting position <ChevronRight size={14} />
          </button>
          <div className={styles.modelNote}>
            <strong>Franka Emika Panda</strong>
            <span>7 joints · parallel gripper · scripted controller</span>
            <a
              href={
                scene?.source ??
                "https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda"
              }
              target="_blank"
              rel="noreferrer"
            >
              Model: MuJoCo Menagerie <ArrowUpRight size={11} />
            </a>
            <a href="/api/demo/license" target="_blank" rel="noreferrer">
              Apache 2.0 license
            </a>
          </div>
        </aside>
      </div>
      <footer className={styles.footer}>
        <span>
          SIMULATION DEMO <i /> Contact-driven grasping
        </span>
        <span>
          Rigid-body dynamics in Python · interactive rendering in Three.js
        </span>
      </footer>
    </main>
  );
}
