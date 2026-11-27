"use client";

import { Box, Check, Download, Package } from "lucide-react";
import type { LibraryObject, TaskState } from "./types";
import styles from "./demo.module.css";

export default function ObjectLibrary({
  objects,
  state,
  busy,
  load,
}: {
  objects: LibraryObject[];
  state: TaskState;
  busy: boolean;
  load: (object: LibraryObject) => Promise<void>;
}) {
  const locked =
    busy ||
    Boolean(state.recording) ||
    Boolean(state.replay) ||
    state.status === "running";
  return (
    <section className={styles.objectLibrary} aria-label="Object library">
      <p>
        Real object scans at their original size. Loading an object starts a
        fresh task.
      </p>
      {objects.map((object) => (
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
      ))}
      {(state.recording || state.replay || state.status === "running") && (
        <p>
          Pause playback and finish the active recording before changing
          objects.
        </p>
      )}
      <p className={styles.objectPhysicsNote}>
        Scans retain their textures and metric scale. The can uses a cylindrical
        collision shape; the toy uses a box. Mass and friction are assigned
        simulation values. The toy is one rigid body.
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
