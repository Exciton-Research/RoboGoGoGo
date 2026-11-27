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
  Undo2,
} from "lucide-react";
import { request } from "../api";
import TaskScene from "./TaskScene";
import {
  ManualControls,
  RecordingControls,
  EpisodeControls,
} from "./EnvironmentControls";
import type { SceneData, TaskState } from "./types";
import styles from "./demo.module.css";

export default function PickPlaceDemo() {
  const [scene, setScene] = useState<SceneData | null>(null);
  const [state, setState] = useState<TaskState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState(0);
  const [seed, setSeed] = useState(0);
  const [connected, setConnected] = useState(false);
  const [tab, setTab] = useState<"task" | "manual" | "episodes">("task");
  function accept(next: TaskState) {
    setState((previous) =>
      !previous ||
      previous.session_id !== next.session_id ||
      next.revision >= previous.revision
        ? next
        : previous,
    );
  }
  useEffect(() => {
    let active = true,
      timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const scene = await request<SceneData>("/demo/scene");
        if (!active) return;
        setScene(scene);
        accept(scene.state);
        setConnected(true);
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
        accept(next);
        setConnected(true);
      } catch {
        if (active) setConnected(false);
      }
      if (active) timer = setTimeout(() => void poll(), 60);
    }
    void load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  async function act(path: string, body: unknown) {
    setBusy(true);
    setError("");
    try {
      const next = await request<TaskState>(`/demo${path}`, body);
      accept(next);
      setConnected(true);
      if (
        path.startsWith("/replay/") ||
        (path === "/replay" && (body as { command: string }).command !== "exit")
      )
        setTab("episodes");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Command failed");
    } finally {
      setBusy(false);
    }
  }
  const running = state?.status === "running";
  const finished = state?.status === "complete" || state?.status === "failed";
  const replaying = Boolean(state?.replay);
  const disabled = busy || !state || !connected;
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a href="/" className={styles.brand}>
          robogogogo<span>.</span>
        </a>
        <div className={styles.breadcrumb}>
          ROBOT MANIPULATION <ChevronRight size={12} /> ENVIRONMENT 01
        </div>
        {scene?.legacy_workspace ? (
          <a className={styles.workspaceLink} href="/simulation/">
            Robot workspace <ArrowUpRight size={14} />
          </a>
        ) : (
          <span className={styles.workspaceLink}>
            Manual control · recording · replay
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
            <p className={styles.eyebrow}>PANDA WORKBENCH</p>
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
            {state?.replay
              ? "Replay · read only"
              : state?.status === "complete"
                ? "Placement complete"
                : state?.status === "failed"
                  ? "Task stopped"
                  : state?.status === "running"
                    ? "Task in progress"
                    : state?.status === "paused"
                      ? state.mode === "manual"
                        ? "Manual control"
                        : "Paused"
                      : state
                        ? "Ready when you are"
                        : "Preparing simulation"}
            <span>{state ? `${state.time.toFixed(1)} s` : "—"}</span>
          </div>
          <div className={styles.controls}>
            <button
              className={styles.runButton}
              disabled={disabled || (finished && Boolean(state?.recording))}
              onClick={() =>
                replaying
                  ? void act("/replay", { command: "exit" })
                  : void act("/command", {
                      command: running ? "pause" : finished ? "reset" : "run",
                      seed,
                    })
              }
            >
              {replaying ? (
                <Undo2 size={16} />
              ) : running ? (
                <Pause size={16} />
              ) : finished ? (
                <RotateCcw size={16} />
              ) : (
                <Play size={16} fill="currentColor" />
              )}
              {busy
                ? "Applying…"
                : replaying
                  ? "Return to live control"
                  : running
                    ? "Pause reference"
                    : finished
                      ? "Reset task"
                      : state?.status === "paused" && state.mode === "script"
                        ? "Resume reference"
                        : "Run reference task"}
            </button>
            <button
              className={styles.resetButton}
              aria-label="Reset task"
              title="Reset task"
              disabled={busy || !state || replaying || Boolean(state.recording)}
              onClick={() => void act("/reset", { seed })}
            >
              <RotateCcw size={17} />
            </button>
          </div>
          {error && (
            <div className={styles.error} role="alert">
              <strong>
                {scene ? "Action not applied" : "Demo unavailable"}
              </strong>
              <p>{error}</p>
              <button onClick={() => setError("")}>Dismiss</button>
              {!scene && <code>robogogogo-demo --setup</code>}
            </div>
          )}
          {!connected && state && (
            <p className={styles.controlNote} role="status">
              Connection interrupted. Reconnecting…
            </p>
          )}
          {state && (
            <>
              {!state.replay && (
                <div className={styles.seedControl}>
                  <label htmlFor="reset-seed">Reset seed</label>
                  <input
                    id="reset-seed"
                    type="number"
                    min={0}
                    max={1000000}
                    step={1}
                    value={seed}
                    disabled={busy || Boolean(state.recording)}
                    onChange={(event) =>
                      setSeed(
                        Math.max(
                          0,
                          Math.min(
                            1000000,
                            Math.trunc(Number(event.target.value)),
                          ),
                        ),
                      )
                    }
                  />
                </div>
              )}
              <RecordingControls
                state={state}
                busy={busy || !connected}
                act={act}
              />
              <div
                className={styles.panelTabs}
                role="tablist"
                aria-label="Environment controls"
              >
                {(["task", "manual", "episodes"] as const).map((name) => (
                  <button
                    key={name}
                    role="tab"
                    aria-selected={tab === name}
                    onClick={() => setTab(name)}
                  >
                    {name === "episodes"
                      ? "Episodes"
                      : name === "manual"
                        ? "Manual"
                        : "Task"}
                  </button>
                ))}
              </div>
              {tab === "manual" && (
                <ManualControls
                  key={`${state.episode_id}:${state.mode}`}
                  state={state}
                  busy={busy || !connected}
                  act={act}
                />
              )}
              {tab === "episodes" && (
                <EpisodeControls
                  state={state}
                  busy={busy || !connected}
                  act={act}
                />
              )}
              {tab === "task" && (
                <div className={styles.taskDetail}>
                  <p>{state.message}</p>
                  <ol className={styles.steps}>
                    {[
                      "Approach & align",
                      "Establish grasp",
                      "Lift & carry",
                      "Place & release",
                      "Withdraw & verify",
                    ].map((label, index) => {
                      const group =
                        state.phase < 2
                          ? 0
                          : state.phase === 2
                            ? 1
                            : state.phase < 5
                              ? 2
                              : state.phase < 7
                                ? 3
                                : 4;
                      const done =
                        state.mode === "script" &&
                        (state.status === "complete" || group > index);
                      return (
                        <li
                          key={label}
                          data-active={
                            state.mode === "script" && group === index && !done
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
                          <div>{label}</div>
                        </li>
                      );
                    })}
                  </ol>
                </div>
              )}
            </>
          )}
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
          <div className={styles.modelNote}>
            <strong>Franka Emika Panda</strong>
            <span>7 joints · parallel gripper · 20 Hz control</span>
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
          PANDA ENVIRONMENT <i /> Contact-driven grasping
        </span>
        <span>Explicit simulation steps · synchronized episode recording</span>
      </footer>
    </main>
  );
}
