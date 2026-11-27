"use client";

import { useEffect, useRef, useState } from "react";
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
import ObjectLibrary from "./ObjectLibrary";
import {
  ManualControls,
  RecordingControls,
  EpisodeControls,
} from "./EnvironmentControls";
import type {
  LibraryObject,
  PerceptionMode,
  SceneData,
  TaskState,
} from "./types";
import styles from "./demo.module.css";

export default function PickPlaceDemo() {
  const [scene, setScene] = useState<SceneData | null>(null);
  const [state, setState] = useState<TaskState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState(0);
  const [seed, setSeed] = useState(0);
  const [perception, setPerception] = useState<PerceptionMode>("simulator");
  const [query, setQuery] = useState("");
  const perceptionSession = useRef("");
  const [connected, setConnected] = useState(false);
  const [tab, setTab] = useState<"objects" | "task" | "manual" | "episodes">(
    "objects",
  );
  const sceneKey = useRef("");
  const accepted = useRef<{ session: string; revision: number } | null>(null);
  function accept(next: TaskState) {
    if (
      accepted.current?.session === next.session_id &&
      accepted.current.revision > next.revision
    )
      return false;
    accepted.current = { session: next.session_id, revision: next.revision };
    setState((previous) =>
      !previous ||
      previous.session_id !== next.session_id ||
      next.revision >= previous.revision
        ? next
        : previous,
    );
    return true;
  }
  async function synchronize(next: TaskState) {
    if (next.scene_id !== sceneKey.current) {
      const updated = await request<SceneData>("/demo/scene");
      if (accept(updated.state)) {
        sceneKey.current = updated.scene_id;
        setScene(updated);
      }
    } else accept(next);
  }
  async function loadObject(object: LibraryObject) {
    setBusy(true);
    setError("");
    try {
      if (!object.ready)
        await request(`/demo/objects/${object.id}/download`, {});
      await synchronize(
        await request<TaskState>("/demo/objects/select", {
          identifier: object.id,
          seed,
        }),
      );
      setPerception(
        state?.pretrained_ready
          ? "vlm"
          : object.id === "block" || object.reference_matching === false
            ? "simulator"
            : "camera",
      );
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Object could not be loaded",
      );
    } finally {
      setBusy(false);
    }
  }
  async function loadClutter(identifier = "soup-can") {
    setBusy(true);
    setError("");
    try {
      for (const object of scene?.objects ?? []) {
        if (object.source && !object.ready)
          await request(`/demo/objects/${object.id}/download`, {});
      }
      await synchronize(
        await request<TaskState>("/demo/objects/select", {
          identifier,
          clutter: true,
          seed,
        }),
      );
      setPerception(
        state?.pretrained_ready
          ? "vlm"
          : identifier === "sugar-box"
            ? "simulator"
            : "camera",
      );
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Table could not be loaded",
      );
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    let active = true,
      timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const scene = await request<SceneData>("/demo/scene");
        if (!active) return;
        setScene(scene);
        sceneKey.current = scene.scene_id;
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
        await synchronize(next);
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
      await synchronize(next);
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
  useEffect(() => {
    if (state && perceptionSession.current !== state.session_id) {
      perceptionSession.current = state.session_id;
      setPerception(
        state.controller_perception ??
          (state.pretrained_ready
            ? "vlm"
            : state.object.id === "block"
              ? "simulator"
              : "camera"),
      );
    }
    if (
      state?.mode === "script" &&
      state.controller_perception &&
      (state.status === "running" || state.status === "paused")
    ) {
      setPerception(state.controller_perception);
      if (state.vision?.query) setQuery(state.vision.query);
    }
  }, [
    state?.mode,
    state?.controller_perception,
    state?.status,
    state?.session_id,
    state?.pretrained_ready,
    state?.vision?.query,
  ]);
  useEffect(() => {
    if (state) setQuery(state.vision?.query ?? state.object.name);
  }, [state?.episode_id]);
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
            <span /> {state?.object.name ?? "Object"} <i /> Placement target
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
              Pick up the {state?.object.name.toLowerCase() ?? "object"} and
              place it inside the marked square.
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
                ? state.layout === "clutter"
                  ? "Target collected"
                  : "Placement complete"
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
                      perception,
                      query: perception === "vlm" ? query : undefined,
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
                        : state?.layout === "clutter"
                          ? perception !== "simulator"
                            ? "Find & collect target"
                            : "Collect target"
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
                  {state.layout === "clutter" && (
                    <button
                      disabled={
                        busy ||
                        Boolean(state.recording) ||
                        state.status === "running"
                      }
                      onClick={() => {
                        const nextSeed = (state.seed % 1000000) + 1;
                        setSeed(nextSeed);
                        void act("/reset", { seed: nextSeed });
                      }}
                    >
                      Shuffle table
                    </button>
                  )}
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
                {(["objects", "task", "manual", "episodes"] as const).map(
                  (name) => (
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
                          : name === "objects"
                            ? "Objects"
                            : "Task"}
                    </button>
                  ),
                )}
              </div>
              {tab === "objects" && scene && (
                <ObjectLibrary
                  objects={scene.objects}
                  state={state}
                  busy={busy || !connected}
                  load={loadObject}
                  loadClutter={loadClutter}
                  selectTarget={(identifier) => act("/target", { identifier })}
                  perception={perception}
                  setPerception={setPerception}
                  query={query}
                  setQuery={setQuery}
                  find={() =>
                    act("/vision/find", {
                      query,
                      expected_revision: state.revision,
                    })
                  }
                />
              )}
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
              : "The fingers hold the object through contact and friction. Opening them releases it under gravity."}
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
