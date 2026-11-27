"use client";

import { useState } from "react";
import { Box, Check, Download, Package } from "lucide-react";
import type { LibraryObject, TaskState } from "./types";
import styles from "./demo.module.css";

export default function ObjectLibrary({
  objects,
  state,
  busy,
  load,
  loadClutter,
  selectTarget,
  cameraMatching,
  setCameraMatching,
}: {
  objects: LibraryObject[];
  state: TaskState;
  busy: boolean;
  load: (object: LibraryObject) => Promise<void>;
  loadClutter: () => Promise<void>;
  selectTarget: (identifier: string) => Promise<void>;
  cameraMatching: boolean;
  setCameraMatching: (enabled: boolean) => void;
}) {
  const [search, setSearch] = useState("");
  const clutter = state.layout === "clutter";
  const locked =
    busy ||
    Boolean(state.recording) ||
    Boolean(state.replay) ||
    state.status === "running";
  return (
    <section className={styles.objectLibrary} aria-label="Object library">
      <div className={styles.clutterControl}>
        <button disabled={locked} onClick={() => void loadClutter()}>
          {busy
            ? "Preparing…"
            : clutter
              ? "Reset cluttered table"
              : "Load cluttered table · 8 objects"}
        </button>
        <p>
          {clutter
            ? "Choose a named target below, then find and collect it. Other objects remain on the table."
            : "Load all eight scans together, or choose one object from the library."}
        </p>
      </div>
      {state.object.id !== "block" && (
        <>
          <label className={styles.visionToggle}>
            <input
              type="checkbox"
              checked={cameraMatching}
              disabled={
                locked || (state.mode === "script" && state.status === "paused")
              }
              onChange={(event) => setCameraMatching(event.target.checked)}
            />
            Find target from camera images
          </label>
          <p className={styles.poseNotice}>
            {cameraMatching
              ? "Inspection RGB + depth · library image matching · scripted grasp. Stops on uncertain matches."
              : "Simulator reference · uses known object poses."}
          </p>
        </>
      )}
      {state.vision && (
        <figure className={styles.visionResult}>
          <img
            src={state.vision.image_url}
            width={640}
            height={384}
            alt={`Inspection camera detection: ${state.vision.name}`}
          />
          <figcaption>
            Found {state.vision.name} · image match{" "}
            {state.vision.score.toFixed(2)}
            <small>
              Catalog similarity score, not a probability. Pickup estimated from
              depth and known upright dimensions.
            </small>
          </figcaption>
        </figure>
      )}
      <input
        className={styles.objectSearch}
        aria-label="Find object by name"
        placeholder="Find an object by name…"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />
      {clutter ? (
        <div className={styles.targetList} aria-label="Objects on the table">
          {state.instances
            .filter((item) =>
              item.name.toLowerCase().includes(search.toLowerCase()),
            )
            .map((item) => (
              <button
                key={item.id}
                disabled={locked || item.selected || item.at_collection}
                data-selected={item.selected}
                onClick={() => void selectTarget(item.id)}
              >
                <span>{item.name}</span>
                <small>
                  {item.at_collection
                    ? "Collected"
                    : item.selected
                      ? "Target"
                      : "Choose target"}
                </small>
              </button>
            ))}
        </div>
      ) : (
        objects
          .filter((object) =>
            object.name.toLowerCase().includes(search.toLowerCase()),
          )
          .map((object) => (
            <article
              key={object.id}
              className={styles.objectCard}
              data-selected={state.object.id === object.id}
            >
              <div className={styles.objectCardHeading}>
                <span className={styles.objectIcon}>
                  {object.category === "Household" ? (
                    <Package size={21} />
                  ) : (
                    <Box size={21} />
                  )}
                </span>
                <div>
                  <span>
                    {object.category}
                    {object.source ? " · YCB scan" : ""}
                  </span>
                  <h3>{object.name}</h3>
                </div>
                {state.object.id === object.id && (
                  <Check size={15} aria-label="Selected object" />
                )}
              </div>
              {object.dimensions && (
                <p>
                  {object.dimensions
                    .map((value) => (value * 100).toFixed(1))
                    .join(" × ")}{" "}
                  cm
                  {object.mass !== undefined &&
                    ` · ${Math.round(object.mass * 1000)} g`}
                </p>
              )}
              <div className={styles.objectCardActions}>
                <button
                  disabled={locked || state.object.id === object.id}
                  onClick={() => void load(object)}
                >
                  {!object.ready && <Download size={12} />}
                  {state.object.id === object.id
                    ? "In scene"
                    : object.ready
                      ? "Load object"
                      : "Download & load"}
                </button>
                {object.source && (
                  <a href={object.source} target="_blank" rel="noreferrer">
                    Source
                  </a>
                )}
              </div>
            </article>
          ))
      )}
      {(state.recording || state.replay || state.status === "running") && (
        <p>
          Pause playback and finish the active recording before changing
          objects.
        </p>
      )}
      <p className={styles.objectPhysicsNote}>
        Scans retain their textures and metric scale. Boxes/cylinders use fitted
        collision shapes; fruit and the meat can use convex mesh collisions.
        Mass and friction are assigned simulation values. The toy is one rigid
        body.
      </p>
      <p className={styles.objectCredit}>
        Scans: YCB Object and Model Set, Calli et al. ·{" "}
        <a
          href="https://creativecommons.org/licenses/by/4.0/"
          target="_blank"
          rel="noreferrer"
        >
          CC BY 4.0
        </a>
      </p>
    </section>
  );
}
