"use client";

import { useState } from "react";
import ImportPreview from "./ImportPreview";
import type { LibraryObject, ObjectImportPreview } from "./types";
import styles from "./demo.module.css";

export default function ObjectImporter({
  objects,
  locked,
  load,
  loadClutter,
  importObject,
}: {
  objects: LibraryObject[];
  locked: boolean;
  load: (object: LibraryObject) => Promise<void>;
  loadClutter: (identifier: string) => Promise<void>;
  importObject: (draft: string, clutter: boolean) => Promise<boolean>;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [size, setSize] = useState("12");
  const [up, setUp] = useState("auto");
  const [turn, setTurn] = useState("0");
  const [mass, setMass] = useState("150");
  const [credit, setCredit] = useState("");
  const [preview, setPreview] = useState<ObjectImportPreview | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const disabled = locked || pending;
  async function prepare(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setPending(true);
    setError("");
    setPreview(null);
    try {
      const form = new FormData();
      for (const [key, value] of Object.entries({
        name,
        size_cm: size,
        up,
        turn,
        mass_g: mass,
        credit,
      }))
        form.set(key, value);
      form.set("file", file);
      const response = await fetch("/api/demo/objects/import/preview", {
        method: "POST",
        body: form,
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : "Check the import settings and try again.",
        );
      setPreview(result);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Import failed");
    } finally {
      setPending(false);
    }
  }
  return (
    <div className={styles.importer}>
      <details>
        <summary>Import a textured object</summary>
        <p>
          GLB with embedded textures, or ZIP containing one OBJ, its MTL and
          textures. Up to 32 MB. Saved in your local library.
        </p>
        <form onSubmit={prepare} onChange={() => setPreview(null)}>
          <fieldset disabled={disabled}>
            <label>
              Object file
              <input
                type="file"
                accept=".glb,.zip"
                required
                onChange={(e) => {
                  const selected = e.target.files?.[0] ?? null;
                  setFile(selected);
                  if (selected)
                    setName(
                      selected.name
                        .replace(/\.[^.]+$/, "")
                        .replace(/[_-]+/g, " "),
                    );
                }}
              />
            </label>
            <label>
              Object name
              <input
                value={name}
                maxLength={120}
                required
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Cracker box"
              />
            </label>
            <label>
              Longest dimension (cm)
              <input
                type="number"
                value={size}
                min="2"
                max="22"
                step="0.1"
                required
                onChange={(e) => setSize(e.target.value)}
              />
            </label>
            <details>
              <summary>Orientation, mass and credit</summary>
              <label>
                Source up axis
                <select value={up} onChange={(e) => setUp(e.target.value)}>
                  {["auto", "X", "Y", "Z", "-X", "-Y", "-Z"].map((axis) => (
                    <option key={axis} value={axis}>
                      {axis === "auto" ? "Auto · GLB Y / OBJ Z" : axis}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Turn after alignment (degrees)
                <input
                  type="number"
                  value={turn}
                  min="-180"
                  max="180"
                  required
                  onChange={(e) => setTurn(e.target.value)}
                />
              </label>
              <label>
                Assigned mass (g)
                <input
                  type="number"
                  value={mass}
                  min="5"
                  max="500"
                  required
                  onChange={(e) => setMass(e.target.value)}
                />
              </label>
              <label>
                Source / license credit
                <input
                  value={credit}
                  maxLength={300}
                  onChange={(e) => setCredit(e.target.value)}
                />
              </label>
            </details>
            <button type="submit" disabled={!file || !name.trim()}>
              {pending ? "Preparing object…" : "Preview object"}
            </button>
          </fieldset>
        </form>
        {error && <p role="alert">{error}</p>}
        {preview && (
          <>
            <ImportPreview preview={preview} />
            <p>
              {preview.object.dimensions
                ?.map((v) => (v * 100).toFixed(1))
                .join(" × ")}{" "}
              cm · {Math.round((preview.object.mass ?? 0) * 1000)} g
            </p>
            <p>
              Rigid convex collision proxy; assigned mass and friction. Narrow
              side is aligned to the gripper before your turn adjustment.
            </p>
            {!preview.collectable && (
              <p role="status">
                This orientation exceeds the current upright pickup range.
                Adjust size or orientation for automatic collection.
              </p>
            )}
            <div className={styles.objectCardActions}>
              <button
                disabled={disabled}
                onClick={() =>
                  void importObject(preview.draft_id, false).then((ok) => {
                    if (ok) setPreview(null);
                  })
                }
              >
                Load imported object
              </button>
              <button
                disabled={disabled}
                onClick={() =>
                  void importObject(preview.draft_id, true).then((ok) => {
                    if (ok) setPreview(null);
                  })
                }
              >
                Load with 8 distractors
              </button>
            </div>
          </>
        )}
      </details>
      {objects.some((object) => object.imported) && (
        <div aria-label="Local imported objects">
          <h3>Your imported objects</h3>
          {objects
            .filter((object) => object.imported)
            .map((object) => (
              <article key={object.id} className={styles.objectCard}>
                <strong>{object.name}</strong>
                {object.credit && <p>{object.credit}</p>}
                <div className={styles.objectCardActions}>
                  <button disabled={disabled} onClick={() => void load(object)}>
                    Load object
                  </button>
                  <button
                    disabled={disabled}
                    onClick={() => void loadClutter(object.id)}
                  >
                    Load with clutter
                  </button>
                </div>
              </article>
            ))}
        </div>
      )}
    </div>
  );
}
