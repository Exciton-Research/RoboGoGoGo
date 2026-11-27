"use client";

import { useState } from "react";
import { Box, Check, Download, Package } from "lucide-react";
import type { LibraryObject, PerceptionMode, TaskState } from "./types";
import ObjectImporter from "./ObjectImporter";
import styles from "./demo.module.css";

export default function ObjectLibrary({
  objects,
  state,
  busy,
  load,
  loadClutter,
  importObject,
  selectTarget,
  collectRemaining,
  perception,
  setPerception,
  query,
  setQuery,
  find,
}: {
  objects: LibraryObject[];
  state: TaskState;
  busy: boolean;
  load: (object: LibraryObject) => Promise<void>;
  loadClutter: (identifier?: string) => Promise<void>;
  importObject: (draft: string, clutter: boolean) => Promise<boolean>;
  selectTarget: (identifier: string) => Promise<void>;
  collectRemaining: () => Promise<void>;
  perception: PerceptionMode;
  setPerception: (mode: PerceptionMode) => void;
  query: string;
  setQuery: (query: string) => void;
  find: () => Promise<void>;
}) {
  const [search, setSearch] = useState("");
  const clutter = state.layout === "clutter";
  const newObjectScene =
    clutter && state.instances.some((item) => item.id === "sugar-box");
  const importedScene =
    clutter && state.instances.some((item) => item.id.startsWith("import-"));
  const locked =
    busy ||
    Boolean(state.recording) ||
    Boolean(state.replay) ||
    state.status === "running";
  return (
    <section className={styles.objectLibrary} aria-label="Object library">
      <ObjectImporter
        objects={objects}
        locked={locked}
        load={load}
        loadClutter={loadClutter}
        importObject={importObject}
      />
      <div className={styles.clutterControl}>
        <button disabled={locked} onClick={() => void loadClutter()}>
          {busy
            ? "Preparing…"
            : newObjectScene
              ? "Load original table · 8 objects"
              : clutter
                ? "Load original table · 8 objects"
                : "Load cluttered table · 8 objects"}
        </button>
        <button disabled={locked} onClick={() => void loadClutter("sugar-box")}>
          {newObjectScene
            ? "Load full table · 9 objects"
            : "New household object · sugar carton"}
        </button>
        <p>
          {clutter
            ? importedScene
              ? "Your imported object alongside eight built-in objects. Shuffle the table, then describe the target for OWLv2."
              : newObjectScene
                ? "Sugar carton alongside all eight original objects. Shuffle the table, then try ‘tall yellow box’."
                : "Choose a named target below, then find and collect it. Other objects remain on the table."
            : "Load the original eight-object table, the new sugar carton scene, or one object."}
        </p>
      </div>
      {clutter && perception === "vlm" && (
        <div className={styles.clutterControl}>
          <button
            disabled={
              locked ||
              state.collection?.active ||
              state.collection?.completed === state.collection?.total
            }
            onClick={() => void collectRemaining()}
          >
            Collect remaining · record
          </button>
          <p>
            {state.collection?.completed ?? 0} / {state.instances.length}{" "}
            collected. Finds each default name with OWLv2 and saves one episode
            per object. Stops on uncertain detection.
          </p>
        </div>
      )}
      <div className={styles.detectorControls}>
        <label htmlFor="object-localization">Find objects with</label>
        <select
          id="object-localization"
          value={perception}
          disabled={
            locked || (state.mode === "script" && state.status === "paused")
          }
          onChange={(event) =>
            setPerception(event.target.value as PerceptionMode)
          }
        >
          <option value="vlm" disabled={!state.pretrained_ready}>
            Pretrained OWLv2{!state.pretrained_ready ? " · setup required" : ""}
          </option>
          <option
            value="camera"
            disabled={
              state.object.id === "block" ||
              state.object.reference_matching === false
            }
          >
            Catalog image baseline
          </option>
          <option value="simulator">Simulator-pose reference</option>
        </select>
        {state.object.reference_matching === false && (
          <p className={styles.poseNotice}>
            New to this environment. No catalog image references; use pretrained
            detection.
          </p>
        )}
        <p className={styles.poseNotice}>
          {perception === "vlm"
            ? "Text + RGB detection · depth-based pickup · no catalog images or dimensions. Stops on uncertain matches."
            : perception === "camera"
              ? "Catalog image matching + depth · known object dimensions."
              : "Simulator reference · uses known object poses."}
        </p>
        {perception === "vlm" && (
          <>
            <label htmlFor="object-description">
              Describe the selected target
            </label>
            <input
              id="object-description"
              value={query}
              maxLength={120}
              placeholder="e.g. yellow citrus fruit"
              disabled={
                locked || (state.mode === "script" && state.status === "paused")
              }
              onChange={(event) => setQuery(event.target.value)}
            />
            <button
              disabled={locked || !query.trim() || state.ended}
              onClick={() => void find()}
            >
              Find in image
            </button>
          </>
        )}
      </div>
      {state.vision && (
        <figure className={styles.visionResult}>
          <img
            src={state.vision.image_url}
            width={640}
            height={384}
            alt={`Inspection camera detection: ${state.vision.name}`}
          />
          <figcaption>
            Detected {state.vision.name} ·{" "}
            {state.vision.model ? "OWLv2 score" : "image match"}{" "}
            {state.vision.score.toFixed(2)}
            <small>
              {state.vision.camera_view && (
                <span>
                  {state.vision.inspection_views
                    ? `Close-up inspection agreed: ${state.vision.inspection_views.join(" + ")}. `
                    : state.vision.camera_view === "front"
                      ? "Front camera · workspace crop after side-view rejection. "
                      : "Side camera. "}
                </span>
              )}
              {state.vision.model
                ? "Pretrained model score; not calibrated certainty. Pickup estimated from depth."
                : "Catalog similarity score, not a probability. Pickup uses known upright dimensions."}
              {state.vision.pickup_error && (
                <span> {state.vision.pickup_error}</span>
              )}
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
          .filter(
            (object) =>
              !object.imported &&
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
        Built-in scans retain their textures and metric scale. Boxes/cylinders
        use fitted collision shapes; fruit and the meat can use convex mesh
        collisions. Mass and friction are assigned simulation values. The toy is
        one rigid body.
      </p>
      <p className={styles.objectCredit}>
        Built-in scans: YCB Object and Model Set, Calli et al. ·{" "}
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
