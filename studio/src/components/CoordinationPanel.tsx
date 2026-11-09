import { useEffect, useState } from "react";
import { Check, Pause, Play, Route } from "lucide-react";
import type { Coordination, GripParameters } from "../types";

export function CoordinationPanel({
  task,
  materials,
  manipulation,
  armCount,
  disabled,
  running,
  onPlan,
  onRun,
  onPause,
}: {
  task: Coordination | null;
  materials: Record<string, import("../types").MaterialParameters>;
  manipulation: import("../types").ManipulationState | null;
  armCount: number;
  disabled: boolean;
  running: boolean;
  onPlan: (
    kind: "inspection" | "handover" | "material_grasp",
    grip: GripParameters,
  ) => void;
  onRun: () => void;
  onPause: () => void;
}) {
  const [kind, setKind] = useState<
    "inspection" | "handover" | "material_grasp"
  >("handover");
  const [grip, setGrip] = useState<GripParameters>({
    object_material: "rigid_reference",
    structural_dynamics: false,
    recovery_behaviors: true,
    adaptive_grasp: true,
    initial_grip_force_n: 24,
    object_damage_force_n: 55,
    grip_force_n: 40,
    friction_static: 0.8,
    friction_dynamic: 0.6,
  });
  const activeMaterial = manipulation?.object.material.name;
  const activeStructural = !!manipulation?.object.structure;
  useEffect(() => {
    if (activeMaterial)
      setGrip((g) => ({
        ...g,
        object_material: activeMaterial,
        structural_dynamics: activeStructural,
      }));
  }, [activeMaterial, activeStructural]);
  return (
    <section
      className="coordination-panel"
      aria-label="Coordinated task planner"
    >
      <div className="section-heading">
        <span>TASK PLANNER</span>
        <Route size={15} />
      </div>
      <label className="task-kind">
        Task recipe
        <select
          aria-label="Task recipe"
          disabled={disabled}
          value={kind}
          onChange={(e) =>
            setKind(
              e.target.value as "inspection" | "handover" | "material_grasp",
            )
          }
        >
          <option value="material_grasp">Material grasp → lift</option>
          <option value="handover">Pick → hand over → place</option>
          <option value="inspection">Shared-station inspection</option>
        </select>
      </label>
      <h3>
        {task?.name ??
          (kind === "material_grasp"
            ? "Grasp and lift material specimen"
            : kind === "handover"
              ? "Pick, hand over, and place"
              : "Shared-station inspection")}
      </h3>
      <p>
        {kind === "material_grasp"
          ? "Validate contact, lift the specimen, and verify that no modeled damage occurred before attempting a handover."
          : kind === "handover"
            ? "The giver lifts a bar, secures the receiver's grasp, then releases. The receiver places it on the destination support."
            : "Both arms stage together, take turns inspecting one station, then return."}
      </p>
      {kind !== "inspection" && (
        <fieldset className="grip-parameters" disabled={disabled}>
          <legend>Material & finger contact</legend>
          <label>
            Object material
            <select
              aria-label="Object material"
              value={grip.object_material}
              onChange={(e) => {
                const name = e.target
                  .value as GripParameters["object_material"];
                const preset = materials[name];
                if (name !== "rigid_reference") setKind("material_grasp");
                setGrip({
                  ...grip,
                  object_material: name,
                  structural_dynamics: name !== "rigid_reference",
                  initial_grip_force_n: preset?.initial_force_n ?? 24,
                  object_damage_force_n: preset?.force_threshold_n ?? 55,
                });
              }}
            >
              <option value="rigid_reference">Rigid reference</option>
              {Object.keys(materials).map((name) => (
                <option key={name} value={name}>
                  {name.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
          {grip.object_material !== "rigid_reference" && (
            <label>
              <input
                type="checkbox"
                checked={grip.structural_dynamics}
                onChange={(e) =>
                  setGrip({ ...grip, structural_dynamics: e.target.checked })
                }
              />
              Structural dynamics (experimental)
            </label>
          )}
          {materials[grip.object_material] && (
            <small>
              Illustrative material:{" "}
              {(materials[grip.object_material].modulus_pa / 1e6).toFixed(2)}{" "}
              MPa ·{" "}
              {(materials[grip.object_material].mass_kg * 1000).toFixed(0)} g.
              {grip.structural_dynamics
                ? " Structural bending, tearing and fracture; simulated damage feedback."
                : " Local indentation and surface damage."}
            </small>
          )}
          <label>
            Actuator limit per jaw (N)
            <input
              aria-label="Grip force per jaw"
              type="number"
              min="0"
              max="80"
              step="1"
              value={grip.grip_force_n}
              onChange={(e) =>
                setGrip({ ...grip, grip_force_n: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Static friction μ
            <input
              aria-label="Static friction"
              type="number"
              min="0"
              max="2"
              step="0.05"
              value={grip.friction_static}
              onChange={(e) =>
                setGrip({ ...grip, friction_static: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Sliding friction μ
            <input
              aria-label="Sliding friction"
              type="number"
              min="0"
              max="2"
              step="0.05"
              value={grip.friction_dynamic}
              onChange={(e) =>
                setGrip({ ...grip, friction_dynamic: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Adaptive force feedback
            <input
              type="checkbox"
              checked={grip.adaptive_grasp}
              onChange={(e) =>
                setGrip({
                  ...grip,
                  adaptive_grasp: e.target.checked,
                  recovery_behaviors:
                    e.target.checked && grip.recovery_behaviors,
                })
              }
            />
          </label>
          <label>
            <input
              type="checkbox"
              disabled={!grip.adaptive_grasp || disabled}
              checked={grip.recovery_behaviors && grip.adaptive_grasp}
              onChange={(e) =>
                setGrip({ ...grip, recovery_behaviors: e.target.checked })
              }
            />{" "}
            Coordinated recovery
          </label>
          <small>
            Bring a free arm to a checked grasp, or carry to a nearby registered
            support. Grip and support loads are verified before release. A new
            contact configuration requires replanning.
          </small>
          {grip.adaptive_grasp && (
            <>
              <label>
                Initial force per jaw (N)
                <input
                  aria-label="Initial grip force"
                  type="number"
                  min="1"
                  max="80"
                  value={grip.initial_grip_force_n}
                  onChange={(e) =>
                    setGrip({
                      ...grip,
                      initial_grip_force_n: Number(e.target.value),
                    })
                  }
                />
              </label>
              <label>
                Object force threshold per jaw (N)
                <input
                  aria-label="Object force threshold"
                  type="number"
                  min="1"
                  max="200"
                  value={grip.object_damage_force_n}
                  onChange={(e) =>
                    setGrip({
                      ...grip,
                      object_damage_force_n: Number(e.target.value),
                    })
                  }
                />
              </label>
              <small>
                Calibrate this threshold for your object. Initial force must be
                at most 75% of this threshold. Commands reserve 25% headroom;
                measured overloads stop the task. Material damage is modeled
                separately for deformable presets; calibrate both before
                hardware use.
              </small>
            </>
          )}
          <small>
            Parameters apply on planning. Sliding friction must not exceed
            static friction.
          </small>
        </fieldset>
      )}
      <button
        className="button"
        disabled={
          disabled ||
          armCount !== 2 ||
          (kind !== "inspection" &&
            (grip.friction_dynamic > grip.friction_static ||
              (grip.adaptive_grasp &&
                (grip.initial_grip_force_n <= 0 ||
                  grip.object_damage_force_n <= 0 ||
                  grip.initial_grip_force_n >
                    0.75 * grip.object_damage_force_n))))
        }
        onClick={() => onPlan(kind, grip)}
      >
        Plan coordinated task
      </button>
      {armCount !== 2 && <p>Load the dual-arm rig to use this fixture.</p>}
      <p className="vla-caption">
        Starts a new recorded task from the current settled pose. Save your
        current episode first if needed.
      </p>
      {manipulation && (
        <div className="grasp-status">
          <div className="section-heading">
            <span>OBJECT & GRIPPERS</span>
            <span>{(manipulation.object.mass * 1000).toFixed(0)} g</span>
          </div>
          <p>
            Object holder:{" "}
            <strong>
              {manipulation.object.owner ?? "Released / on support"}
            </strong>
          </p>
          {manipulation.object.material?.parameters && (
            <div className="gripper-row" aria-label="Object material feedback">
              <strong>
                {manipulation.object.material.name.replaceAll("_", " ")}
              </strong>
              <span>
                {manipulation.object.material.fractured
                  ? "FRACTURE DETECTED"
                  : manipulation.object.material.damaged
                    ? "DAMAGE DETECTED"
                    : (manipulation.object.material.risk ?? 0) > 0.55
                      ? "Unloading: material at risk"
                      : "No modeled damage detected"}
              </span>
              {!manipulation.object.structure && (
                <span>
                  Indentation:{" "}
                  {(
                    (manipulation.object.material.maximum_indentation_m ?? 0) *
                    1000
                  ).toFixed(2)}{" "}
                  mm
                </span>
              )}
              {!manipulation.object.structure && (
                <span>
                  Permanent dent:{" "}
                  {(
                    (manipulation.object.material.maximum_permanent_m ?? 0) *
                    1000
                  ).toFixed(2)}{" "}
                  mm
                </span>
              )}
              <span>
                Peak pressure:{" "}
                {(
                  (manipulation.object.material.peak_pressure_pa ?? 0) / 1000
                ).toFixed(1)}{" "}
                kPa
              </span>
              <span>
                Damage index:{" "}
                {(
                  (manipulation.object.material.damage_fraction ?? 0) * 100
                ).toFixed(1)}
                %
              </span>
              <small>
                Simulated material-state feedback. Red surface regions indicate
                modeled damage.
              </small>
            </div>
          )}
          {Object.entries(manipulation.grippers).map(([name, g]) => (
            <div className="gripper-row" key={name}>
              <strong>{name}</strong>
              <span>
                {g.command} · {(g.width * 1000).toFixed(0)} mm
              </span>
              <span>
                {g.slipping
                  ? "SLIPPING"
                  : g.holding
                    ? "Bilateral contact"
                    : "No bilateral contact"}
              </span>
              <span>
                Normal: {g.normal_force_n.map((n) => n.toFixed(1)).join(" / ")}{" "}
                N
              </span>
              <span>
                Friction:{" "}
                {g.friction_force_n.map((n) => n.toFixed(1)).join(" / ")} N
              </span>
              <span>
                Relative speed: {(g.slip_speed_m_s * 1000).toFixed(1)} mm/s
              </span>
            </div>
          ))}
          {manipulation.recovery && (
            <div className="gripper-row" aria-label="Recovery status">
              <strong>
                Recovery · {manipulation.recovery.phase.replaceAll("_", " ")}
              </strong>
              <span>
                {manipulation.recovery.reason?.replaceAll("_", " ") ??
                  "Monitoring grip and material risk"}
              </span>
              <span>
                Attempts {manipulation.recovery.attempts} · Helper{" "}
                {manipulation.recovery.helper ?? "none"}
              </span>
              <span>
                {manipulation.recovery.risk_hold &&
                  "Motion paused for material risk · "}
                Checked edges {manipulation.recovery.search_checks} · Route{" "}
                {manipulation.recovery.route_progress}/
                {manipulation.recovery.route_waypoints}
                {manipulation.recovery.support_index != null &&
                  ` · Support ${manipulation.recovery.support_index + 1}`}
              </span>
              {manipulation.recovery.rejected_candidates.length > 0 && (
                <details>
                  <summary>Candidate search decisions</summary>
                  {manipulation.recovery.rejected_candidates.map(
                    (candidate, index) => (
                      <p key={index}>
                        {candidate.candidate ??
                          (candidate.support != null
                            ? `Support ${candidate.support + 1}`
                            : "Search")}
                        : {candidate.reason}
                      </p>
                    ),
                  )}
                </details>
              )}
              <span>
                Support load{" "}
                {manipulation.support_load_n
                  ?.map((n) => n.toFixed(2))
                  .join(" / ")}{" "}
                N
              </span>
            </div>
          )}
          {manipulation.object.structure && (
            <p>
              Structural risk{" "}
              {((manipulation.object.material.risk ?? 0) * 100).toFixed(0)}% of
              initiation · {manipulation.object.structure.fragments.length}{" "}
              pieces · {manipulation.object.structure.broken_bonds} broken bonds
            </p>
          )}
          {Object.entries(manipulation.adaptive_grasp ?? {}).map(
            ([name, feedback]) => (
              <div className="gripper-row" key={`feedback-${name}`}>
                <strong>
                  {name} · {feedback.status.replaceAll("_", " ")}
                </strong>
                <span>
                  Command: {feedback.force_setpoint_n.toFixed(1)} N / jaw
                </span>
                <span>Peak measured: {feedback.peak_force_n.toFixed(1)} N</span>
                <span>
                  Object threshold: {feedback.damage_force_n.toFixed(1)} N
                </span>
                <span>
                  Filtered slip:{" "}
                  {(feedback.filtered_slip_speed_m_s * 1000).toFixed(1)} mm/s
                </span>
                {feedback.motion_hold && (
                  <span>Arm motion held for grip recovery</span>
                )}
                {feedback.fault && <span role="alert">{feedback.fault}</span>}
              </div>
            ),
          )}
          {manipulation.object.damage_limit_exceeded && (
            <p role="alert">Object force envelope exceeded</p>
          )}
          {manipulation.events.length > 0 && (
            <details>
              <summary>Grasp events ({manipulation.events.length})</summary>
              {manipulation.events.map((event, i) => (
                <p key={i}>
                  {event.time.toFixed(2)} s · {event.event.replaceAll("_", " ")}{" "}
                  · {event.arm ?? event.receiver ?? "verified"}
                </p>
              ))}
            </details>
          )}
        </div>
      )}
      {task && (
        <>
          <div className={`task-status ${task.status}`}>
            <strong>{task.status.toUpperCase()}</strong>
            <span>{task.stage_name}</span>
          </div>
          <ol className="task-stages">
            {task.stages.map((stage, index) => (
              <li
                key={stage.name}
                className={
                  index < task.stage_index
                    ? "complete"
                    : index === task.stage_index
                      ? "current"
                      : ""
                }
              >
                <span className="task-stage-number">
                  {index < task.stage_index ? <Check size={12} /> : index + 1}
                </span>
                <div>
                  <strong>{stage.name}</strong>
                  <small>
                    {stage.owner
                      ? `${task.kind !== "inspection" ? "Assigned arm" : "Station reserved"} · ${stage.owner}`
                      : "Coordinated repositioning"}{" "}
                    ·{" "}
                    {stage.method === "direct"
                      ? "direct path"
                      : stage.method === "cartesian"
                        ? "level Cartesian path"
                        : "RRT detour"}
                  </small>
                </div>
              </li>
            ))}
          </ol>
          <div className="task-clearance">
            <span>Required clearance</span>
            <strong>{(task.margin_m * 1000).toFixed(0)} mm</strong>
            <span>
              {task.kind !== "inspection"
                ? "Minimum effective clearance"
                : "Minimum observed"}
            </span>
            <strong>
              {task.minimum_clearance_m == null
                ? "—"
                : `${(task.minimum_clearance_m * 1000).toFixed(1)} mm`}
            </strong>
          </div>
          {task.reason && <p role="alert">{task.reason}</p>}
          {running ? (
            <button className="button" onClick={onPause}>
              <Pause size={13} />
              Pause after current action
            </button>
          ) : (
            <button
              className="button"
              disabled={
                disabled ||
                !["ready", "paused", "running"].includes(task.status)
              }
              onClick={onRun}
            >
              <Play size={13} />
              {task.status === "paused"
                ? "Resume coordinated task"
                : "Run coordinated task"}
            </button>
          )}
          <p className="vla-caption">
            Arms and payload are checked before each physics substep. Intended
            gripper/object contacts are allowed; payload/support contact permits
            3 mm tolerance.
          </p>
        </>
      )}
      <p className="vla-caption">
        Compliant finger patches with static/sliding friction and force-limited
        jaws. Objects move through contact forces, with reactions on the arms.
        Sampled contact geometry is not a calibrated material or mesh model.
      </p>
    </section>
  );
}
