"use client";

import { useState } from "react";
import { Circle, Download, Pause, Play, Save, Square } from "lucide-react";
import type { TaskState } from "./types";
import styles from "./demo.module.css";

type Act = (path: string, body: unknown) => Promise<void>;
type Props = { state: TaskState; busy: boolean; act: Act };

export function RecordingControls({ state, busy, act }: Props) {
  const [images, setImages] = useState(true);
  const recording = state.recording;
  const ended = state.ended;
  if (state.replay) return null;
  return (
    <section className={styles.recordBar} aria-label="Episode recording">
      {recording ? (
        <>
          <div className={styles.recordingLabel}>
            <span /> Recording · {recording.transitions} steps
          </div>
          <div className={styles.recordActions}>
            <button
              disabled={busy || recording.transitions === 0}
              onClick={() => void act("/record", { command: "save" })}
            >
              <Save size={13} /> Save
            </button>
            <button
              disabled={busy}
              onClick={() => void act("/record", { command: "discard" })}
            >
              Discard
            </button>
          </div>
        </>
      ) : (
        <>
          <button
            disabled={busy || ended}
            onClick={() => void act("/record", { command: "start", images })}
          >
            <Circle size={12} /> Record episode
          </button>
          <label>
            <input
              type="checkbox"
              checked={images}
              onChange={(event) => setImages(event.target.checked)}
            />{" "}
            RGB
          </label>
        </>
      )}
      {state.last_saved && !recording && (
        <p className={styles.savedNotice}>
          Saved {state.last_saved.transitions} steps ·{" "}
          {state.last_saved.outcome}
        </p>
      )}
    </section>
  );
}

export function ManualControls({ state, busy, act }: Props) {
  const [distance, setDistance] = useState(10);
  const [opening, setOpening] = useState(() =>
    Math.round(state.joint_target[7] * 1000),
  );
  const disabled =
    busy || Boolean(state.replay) || state.status === "running" || state.ended;
  function jog(axis: number, direction: number) {
    const delta = [0, 0, 0];
    delta[axis] = (distance / 1000) * direction;
    return act("/manual", { delta, expected_revision: state.revision });
  }
  return (
    <section className={styles.manual} aria-label="Manual robot control">
      <p>
        Move the tool in world coordinates. Motion and gravity advance only when
        you act.
      </p>
      {state.status === "running" && (
        <p className={styles.controlNote}>
          Pause the reference task to take control.
        </p>
      )}
      <div className={styles.controlTitle}>
        Translation step{" "}
        <select
          aria-label="Translation step"
          value={distance}
          onChange={(event) => setDistance(Number(event.target.value))}
        >
          {[5, 10, 20].map((value) => (
            <option key={value} value={value}>
              {value} mm
            </option>
          ))}
        </select>
      </div>
      <div className={styles.jogGrid}>
        {["X", "Y", "Z"].map((axis, index) => (
          <div key={axis}>
            <button
              disabled={disabled}
              aria-label={`Move ${axis} negative`}
              onClick={() => void jog(index, -1)}
            >
              − {axis}
            </button>
            <span>
              {state.tool[index].toFixed(3)}
              <small>m</small>
            </span>
            <button
              disabled={disabled}
              aria-label={`Move ${axis} positive`}
              onClick={() => void jog(index, 1)}
            >
              + {axis}
            </button>
          </div>
        ))}
      </div>
      <div className={styles.controlTitle}>
        Target opening <span>{opening} mm</span>
      </div>
      <input
        className={styles.openingSlider}
        aria-label="Finger opening in millimeters"
        type="range"
        min={0}
        max={80}
        value={opening}
        disabled={disabled}
        onChange={(event) => setOpening(Number(event.target.value))}
      />
      <div className={styles.gripperButtons}>
        <button
          disabled={disabled}
          onClick={() => {
            setOpening(0);
            void act("/manual", {
              opening: 0,
              expected_revision: state.revision,
            });
          }}
        >
          Close
        </button>
        <button
          disabled={disabled}
          onClick={() => {
            setOpening(80);
            void act("/manual", {
              opening: 0.08,
              expected_revision: state.revision,
            });
          }}
        >
          Open
        </button>
        <button
          disabled={disabled}
          onClick={() =>
            void act("/manual", {
              opening: opening / 1000,
              expected_revision: state.revision,
            })
          }
        >
          Apply
        </button>
      </div>
      <button
        className={styles.stepButton}
        disabled={disabled}
        onClick={() =>
          void act("/step", {
            action: state.joint_target,
            expected_revision: state.revision,
          })
        }
      >
        <Square size={11} /> Step 50 ms · hold current target
      </button>
      <p className={styles.controlNote}>
        {state.step_index} control steps · {state.finger_contacts}/2 fingers
        touching the block
      </p>
    </section>
  );
}

export function EpisodeControls({ state, busy, act }: Props) {
  const [seekFrame, setSeekFrame] = useState<number | null>(null);
  const replay = state.replay;
  async function seek(frame: number) {
    await act("/replay", { command: "seek", frame });
    setSeekFrame(null);
  }
  return (
    <section className={styles.episodePanel} aria-label="Recorded episodes">
      {replay && (
        <div className={styles.replayPanel}>
          <div className={styles.controlTitle}>
            Recorded state <span>Read only</span>
          </div>
          <p>
            {replay.rgb ? "RGB + physical state" : "Physical state only"} ·{" "}
            {replay.outcome}
          </p>
          <input
            aria-label="Replay frame"
            type="range"
            min={0}
            max={replay.frames - 1}
            value={seekFrame ?? replay.frame}
            disabled={busy}
            onChange={(event) => setSeekFrame(Number(event.target.value))}
            onPointerUp={(event) =>
              void seek(Number(event.currentTarget.value))
            }
            onKeyUp={(event) => void seek(Number(event.currentTarget.value))}
          />
          <div className={styles.replayTransport}>
            <button
              disabled={busy}
              onClick={() =>
                void act("/replay", {
                  command: replay.playing ? "pause" : "play",
                })
              }
            >
              {replay.playing ? <Pause size={13} /> : <Play size={13} />}
              {replay.playing ? "Pause replay" : "Play replay"}
            </button>
            <span>
              {replay.frame} / {replay.frames - 1}
            </span>
          </div>
        </div>
      )}
      {!state.episodes.length && (
        <p className={styles.emptyEpisodes}>
          Record a manual or reference run. Saved episodes will appear here with
          replay and download controls.
        </p>
      )}
      {state.episodes.map((episode) => (
        <article
          key={episode.id}
          className={styles.episodeRow}
          data-selected={replay?.id === episode.id}
        >
          <div>
            <strong>
              {episode.outcome === "success"
                ? "Successful placement"
                : episode.outcome === "failed"
                  ? "Failed attempt"
                  : "Partial episode"}
            </strong>
            <span>{episode.object_name ?? "Original block"}</span>
            <span>
              {episode.transitions} steps · {episode.duration.toFixed(1)} s ·
              seed {episode.seed}
            </span>
            <small>
              {episode.rgb ? "RGB + state" : "State only"} ·{" "}
              {new Date(episode.created).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </small>
          </div>
          <div className={styles.episodeActions}>
            <button
              disabled={busy || Boolean(state.recording)}
              onClick={() => void act(`/replay/${episode.id}`, {})}
            >
              <Play size={12} /> Replay
            </button>
            <a
              href={`/api/demo/episodes/${episode.id}/download`}
              aria-label={`Download episode ${episode.id}`}
              title="Download episode archive"
            >
              <Download size={14} />
            </a>
          </div>
        </article>
      ))}
    </section>
  );
}
