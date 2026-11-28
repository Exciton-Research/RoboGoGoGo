"use client";

import { useEffect, useRef, useState } from "react";
import { request } from "../api";
import type {
  LibraryObject,
  LayoutLibrary,
  TableLayout,
  TaskState,
} from "./types";
import styles from "./demo.module.css";

export default function SceneEditor({
  objects,
  state,
  locked,
  apply,
}: {
  objects: LibraryObject[];
  state: TaskState;
  locked: boolean;
  apply: (layout: TableLayout) => Promise<boolean>;
}) {
  const [open, setOpen] = useState(false);
  const [library, setLibrary] = useState<LayoutLibrary | null>(null);
  const [draft, setDraft] = useState<TableLayout>({
    name: "My table",
    seed: 0,
    target: "",
    objects: [],
  });
  const [selected, setSelected] = useState("");
  const [addition, setAddition] = useState("");
  const [saved, setSaved] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const drag = useRef<{ id: string; dx: number; dy: number } | null>(null);
  const disabled = locked || pending;
  const byId = new Map(objects.map((object) => [object.id, object]));
  useEffect(() => {
    if (!open) return;
    let active = true;
    request<LayoutLibrary>("/demo/layouts")
      .then((result) => {
        if (active) setLibrary(result);
      })
      .catch((failure) => {
        if (active) setError(String(failure.message));
      });
    return () => {
      active = false;
    };
  }, [open]);
  const bounds = library?.workspace ?? [0.21, 0.81, -0.39, 0.015];
  const [xmin, xmax, ymin, ymax] = bounds;
  const width = (xmax - xmin) * 1000,
    height = (ymax - ymin) * 1000;
  const gap = library?.gap ?? 0.008;
  const problems = new Set<string>();
  for (const [i, item] of draft.objects.entries()) {
    const size = byId.get(item.id)?.dimensions;
    if (!size || !byId.get(item.id)?.ready) {
      problems.add(item.id);
      continue;
    }
    if (
      item.x - size[0] / 2 < xmin - 1e-9 ||
      item.x + size[0] / 2 > xmax + 1e-9 ||
      item.y - size[1] / 2 < ymin - 1e-9 ||
      item.y + size[1] / 2 > ymax + 1e-9
    )
      problems.add(item.id);
    for (const other of draft.objects.slice(0, i)) {
      const otherSize = byId.get(other.id)?.dimensions;
      if (
        otherSize &&
        Math.abs(item.x - other.x) <
          (size[0] + otherSize[0]) / 2 + gap - 1e-9 &&
        Math.abs(item.y - other.y) < (size[1] + otherSize[1]) / 2 + gap - 1e-9
      ) {
        problems.add(item.id);
        problems.add(other.id);
      }
    }
  }
  const invalid =
    !draft.objects.length ||
    !draft.name.trim() ||
    problems.size > 0 ||
    !draft.objects.some((item) => item.id === draft.target);
  const matchesLive =
    draft.target === state.table_layout.target &&
    draft.objects.length === state.table_layout.objects.length &&
    draft.objects.every((item) =>
      state.table_layout.objects.some(
        (live) =>
          live.id === item.id &&
          Math.abs(live.x - item.x) < 1e-8 &&
          Math.abs(live.y - item.y) < 1e-8,
      ),
    );
  function edit(next: TableLayout) {
    setDraft(next);
    setMessage("");
    setError("");
  }
  function move(id: string, x: number, y: number) {
    const size = byId.get(id)?.dimensions ?? [0.05, 0.05];
    const clamp = (v: number, low: number, high: number) =>
      Math.min(high, Math.max(low, v));
    edit({
      ...draft,
      objects: draft.objects.map((item) =>
        item.id === id
          ? {
              ...item,
              x: clamp(x, xmin + size[0] / 2, xmax - size[0] / 2),
              y: clamp(y, ymin + size[1] / 2, ymax - size[1] / 2),
            }
          : item,
      ),
    });
  }
  function point(event: React.PointerEvent<SVGSVGElement>) {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return null;
    return new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
  }
  async function shuffle(next: TableLayout) {
    setPending(true);
    setError("");
    try {
      const result = await request<TableLayout>("/demo/layouts/shuffle", next);
      edit({ ...result, id: next.id });
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Could not arrange these objects",
      );
    } finally {
      setPending(false);
    }
  }
  async function save() {
    setPending(true);
    setError("");
    try {
      const result = await request<TableLayout>("/demo/layouts/save", draft);
      setDraft(result);
      setSaved(result.id ?? "");
      setLibrary(await request<LayoutLibrary>("/demo/layouts"));
      setMessage(
        `Saved “${result.name}” locally. Apply to table to use this arrangement.`,
      );
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Layout could not be saved",
      );
    } finally {
      setPending(false);
    }
  }
  return (
    <details
      className={styles.sceneEditor}
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>Build your own table</summary>
      <p>
        Combine up to nine distinct objects. Drag their footprints, then apply
        the arrangement to the simulation. Editing this plan leaves the live
        table in place.
      </p>
      <div className={styles.editorActions}>
        <button
          disabled={disabled}
          onClick={() => {
            const items = state.table_layout.objects.filter(
              (item) => item.id !== "block",
            );
            const target = items.some(
              (item) => item.id === state.table_layout.target,
            )
              ? state.table_layout.target
              : (items[0]?.id ?? "");
            edit({
              ...state.table_layout,
              objects: items,
              target,
            });
            setSelected(target);
          }}
        >
          Use current table
        </button>
        <button
          disabled={disabled}
          onClick={() => {
            edit({ name: "My table", seed: 0, target: "", objects: [] });
            setSelected("");
            setSaved("");
          }}
        >
          New table
        </button>
      </div>
      <label>
        Table name
        <input
          value={draft.name}
          maxLength={80}
          disabled={disabled}
          onChange={(e) => edit({ ...draft, name: e.target.value })}
        />
      </label>
      <label>
        Add a library object
        <select
          aria-label="Add a library object"
          value={addition}
          disabled={disabled || draft.objects.length >= 9}
          onChange={(e) => setAddition(e.target.value)}
        >
          <option value="">Choose an object…</option>
          {objects
            .filter(
              (object) =>
                object.id !== "block" &&
                !draft.objects.some((item) => item.id === object.id),
            )
            .map((object) => (
              <option
                key={object.id}
                value={object.id}
                disabled={!object.ready || !object.dimensions}
              >
                {object.name}
                {object.imported ? " · imported" : ""}
                {!object.ready ? " · download from library first" : ""}
              </option>
            ))}
        </select>
      </label>
      <button
        disabled={disabled || !addition || draft.objects.length >= 9}
        onClick={() => {
          const next = {
            ...draft,
            target: draft.objects.some((item) => item.id === draft.target)
              ? draft.target
              : addition,
            objects: [...draft.objects, { id: addition, x: 0.5, y: -0.2 }],
          };
          setSelected(addition);
          setAddition("");
          void shuffle(next);
        }}
      >
        Add & arrange
      </button>
      <figure className={styles.layoutPlan}>
        <figcaption>TOP VIEW · collection area ↑</figcaption>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="group"
          aria-label="Table layout editor"
          onPointerMove={(event) => {
            if (!drag.current || disabled) return;
            const p = point(event);
            if (p)
              move(
                drag.current.id,
                xmin + p.x / 1000 + drag.current.dx,
                ymax - p.y / 1000 + drag.current.dy,
              );
          }}
          onPointerUp={(event) => {
            drag.current = null;
            if (event.currentTarget.hasPointerCapture(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
        >
          <rect
            x="0"
            y="0"
            width={width}
            height={height}
            fill="#e8eadf"
            stroke="#a8b59a"
            strokeWidth="3"
          />
          {draft.objects.map((item, index) => {
            const size = byId.get(item.id)?.dimensions ?? [0.05, 0.05];
            const cx = (item.x - xmin) * 1000,
              cy = (ymax - item.y) * 1000;
            return (
              <g
                key={item.id}
                role="button"
                tabIndex={disabled ? -1 : 0}
                aria-disabled={disabled}
                aria-label={`Move ${byId.get(item.id)?.name ?? item.id}`}
                data-selected={selected === item.id}
                className={styles.planObject}
                onPointerDown={(event) => {
                  if (disabled) return;
                  event.preventDefault();
                  setSelected(item.id);
                  const svg = event.currentTarget.ownerSVGElement;
                  const matrix = svg?.getScreenCTM();
                  if (!svg || !matrix) return;
                  const p = new DOMPoint(
                    event.clientX,
                    event.clientY,
                  ).matrixTransform(matrix.inverse());
                  drag.current = {
                    id: item.id,
                    dx: item.x - xmin - p.x / 1000,
                    dy: item.y - ymax + p.y / 1000,
                  };
                  svg.setPointerCapture(event.pointerId);
                }}
                onKeyDown={(event) => {
                  if (disabled) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setSelected(item.id);
                    return;
                  }
                  const step = event.shiftKey ? 0.01 : 0.002;
                  const delta = {
                    ArrowLeft: [-step, 0],
                    ArrowRight: [step, 0],
                    ArrowUp: [0, step],
                    ArrowDown: [0, -step],
                  }[event.key];
                  if (delta) {
                    event.preventDefault();
                    setSelected(item.id);
                    move(item.id, item.x + delta[0], item.y + delta[1]);
                  }
                }}
              >
                <rect
                  x={cx - size[0] * 500}
                  y={cy - size[1] * 500}
                  width={size[0] * 1000}
                  height={size[1] * 1000}
                  rx="4"
                  fill={
                    problems.has(item.id)
                      ? "#f4c5b4"
                      : item.id === draft.target
                        ? "#a9c4a3"
                        : "#fbfaf3"
                  }
                  stroke={selected === item.id ? "#283f31" : "#72846a"}
                  strokeWidth={selected === item.id ? 4 : 2}
                />
                <text
                  x={cx}
                  y={cy}
                  dominantBaseline="central"
                  textAnchor="middle"
                  fontSize="20"
                  fill="#283f31"
                  pointerEvents="none"
                >
                  {index + 1}
                </text>
              </g>
            );
          })}
        </svg>
        <p>
          Drag or use arrow keys: 2 mm · Shift + arrows: 1 cm. Footprints need 8
          mm clearance.
        </p>
        {draft.objects.length > 0 && (
          <p>
            {matchesLive
              ? "Plan matches the live table’s starting positions."
              : "Plan differs from the live table. Apply it, or use the current table to copy its starting positions."}
          </p>
        )}
      </figure>
      <div className={styles.planObjects}>
        {draft.objects.map((item, index) => (
          <div key={item.id}>
            <button
              disabled={disabled}
              data-selected={selected === item.id}
              onClick={() => setSelected(item.id)}
            >
              {index + 1}. {byId.get(item.id)?.name ?? item.id}
            </button>
            <button
              aria-label={`Remove ${byId.get(item.id)?.name ?? item.id}`}
              disabled={disabled}
              onClick={() => {
                const remaining = draft.objects.filter(
                  (other) => other.id !== item.id,
                );
                edit({
                  ...draft,
                  objects: remaining,
                  target:
                    draft.target === item.id
                      ? (remaining[0]?.id ?? "")
                      : draft.target,
                });
                if (selected === item.id) setSelected("");
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
      {draft.objects.find((item) => item.id === selected) && (
        <div className={styles.editorCoordinates}>
          {(["x", "y"] as const).map((axis) => (
            <label key={axis}>
              {axis.toUpperCase()} position (cm)
              <input
                type="number"
                step="0.2"
                disabled={disabled}
                value={Number(
                  (
                    (draft.objects.find((item) => item.id === selected)?.[
                      axis
                    ] ?? 0) * 100
                  ).toFixed(2),
                )}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (!Number.isFinite(value)) return;
                  const item = draft.objects.find(
                    (item) => item.id === selected,
                  )!;
                  move(
                    item.id,
                    axis === "x" ? value / 100 : item.x,
                    axis === "y" ? value / 100 : item.y,
                  );
                }}
              />
            </label>
          ))}
        </div>
      )}
      <label>
        Collection target
        <select
          value={draft.target}
          aria-label="Collection target"
          disabled={disabled}
          onChange={(e) => edit({ ...draft, target: e.target.value })}
        >
          <option value="" disabled>
            Choose target…
          </option>
          {draft.objects.map((item) => (
            <option key={item.id} value={item.id}>
              {byId.get(item.id)?.name ?? item.id}
            </option>
          ))}
        </select>
      </label>
      {problems.size > 0 && (
        <p role="status">
          Highlighted objects overlap, extend outside the pickup area, or need
          their cached assets restored. Move them or shuffle to arrange again.
        </p>
      )}
      <div className={styles.editorActions}>
        <button
          disabled={disabled || !draft.objects.length}
          onClick={() =>
            void shuffle({ ...draft, seed: (draft.seed % 1000000) + 1 })
          }
        >
          Shuffle layout
        </button>
        <button
          disabled={disabled || invalid}
          onClick={() =>
            void apply(draft).then((ok) => {
              if (ok)
                setMessage(
                  `Applied “${draft.name}”. Choose OWLv2 to find and collect the target.`,
                );
            })
          }
        >
          Apply to table
        </button>
        <button disabled={disabled || invalid} onClick={() => void save()}>
          Save layout
        </button>
      </div>
      {pending && <p role="status">Preparing layout…</p>}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
      {Boolean(library?.items.length) && (
        <>
          <label>
            Saved local tables
            <select
              aria-label="Saved local tables"
              value={saved}
              disabled={disabled}
              onChange={(e) => setSaved(e.target.value)}
            >
              <option value="">Choose a saved layout…</option>
              {library?.items.map((layout) => (
                <option key={layout.id} value={layout.id}>
                  {layout.name} · {layout.objects.length} objects
                </option>
              ))}
            </select>
          </label>
          <button
            disabled={disabled || !saved}
            onClick={() => {
              const layout = library?.items.find((item) => item.id === saved);
              if (layout)
                void apply(layout).then((ok) => {
                  if (ok) {
                    setDraft(layout);
                    setSelected(layout.target);
                    setMessage(`Opened “${layout.name}”.`);
                  }
                });
            }}
          >
            Open saved table
          </button>
        </>
      )}
      <p>
        Saved on this computer with your object cache. One instance per asset;
        upright placements only. Automatic grasping still depends on visibility,
        shape and gripper reach.
      </p>
    </details>
  );
}
