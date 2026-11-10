import { useEffect, useState } from "react";
import { MobileDesignControls } from "./MobileDesignControls";
import { VlaPanel } from "../VlaPanel";
import {
  ArrowRight,
  Box,
  Check,
  Circle,
  Crosshair,
  Grid2X2,
  Home,
  Settings2,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";
import type { Vec3 } from "../types";
import { nice } from "../format";
import { Sparkline } from "./Sparkline";

export function Inspector() {
  const {
    snapshot,
    history,
    setError,
    selected,
    inspector,
    setInspector,
    goals,
    setGoals,
    setGoalDirty,
    replay,
    vla,
    disabled,
    command,
    state,
    model,
    selectedJoint,
    selectedJointIndex,
    changed,
  } = useWorkspace();
  const [armName, setArmName] = useState("");
  const activeArm =
    model?.arms.find((arm) => arm.name === armName) ?? model?.arms[0];
  const [target, setTarget] = useState<Vec3>([0, 0, 0]);
  useEffect(() => {
    if (activeArm && snapshot)
      setTarget(snapshot.state.targets[activeArm.name]);
  }, [activeArm?.name, snapshot?.epoch]);
  return (
    <>
      {" "}
      <aside className="inspector">
        <div className="panel-tabs">
          <button
            className={inspector === "task" ? "active" : ""}
            onClick={() => setInspector("task")}
          >
            <Target size={14} />
            Episode
          </button>
          <button
            className={inspector === "control" ? "active" : ""}
            onClick={() => setInspector("control")}
          >
            <SlidersHorizontal size={14} />
            Robot
          </button>
          <button
            className={inspector === "inspect" ? "active" : ""}
            onClick={() => setInspector("inspect")}
          >
            <Settings2 size={14} />
            Inspect
          </button>
        </div>
        <div className="inspector-scroll">
          {inspector === "task" ? (
            <VlaPanel
              episode={snapshot?.episode ?? null}
              model={model ?? null}
              disabled={disabled}
              collecting={vla.collecting}
              onStart={(instruction, variant, rig) =>
                void vla.start(instruction, variant, rig)
              }
              onCapture={() => void vla.capture()}
              onStereo={(left, right) => void vla.analyzeStereo(left, right)}
              stereo={vla.stereo}
              onAct={(delta) => void vla.act(delta)}
              onCollect={() => void vla.collect()}
              onStop={vla.stop}
              onSave={() => void vla.save()}
              onError={setError}
            />
          ) : inspector === "control" ? (
            <>
              <div className="inspector-title">
                <div className="section-kicker">ROBOT CONTROLLER</div>
                <h2>Joint position</h2>
                <p>
                  Debug controls · using these ends the current robot episode.
                </p>
              </div>
              {
                <>
                  <MobileDesignControls />
                  <div className="control-mode">
                    <span className="mode-dot" />
                    Computed torque PD<span className="chip">ACTIVE</span>
                  </div>
                  <div className="task-variant">
                    <label htmlFor="active-arm">Control arm</label>
                    <select
                      id="active-arm"
                      aria-label="Control arm"
                      value={activeArm?.name ?? ""}
                      onChange={(e) => setArmName(e.target.value)}
                    >
                      {model?.arms.map((arm) => (
                        <option key={arm.name} value={arm.name}>
                          {arm.name} · {arm.joint_indices.length} joints
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="joint-list">
                    {model?.joints.map((joint, i) =>
                      activeArm?.joint_indices.includes(i) ? (
                        <div className="joint-control" key={joint.name}>
                          <div className="joint-heading">
                            <label htmlFor={`joint-${i}`}>
                              <span>J{i + 1}</span>
                              {joint.name.replaceAll("_", " ")}
                            </label>
                            <div className="joint-number">
                              <input
                                aria-label={`${joint.name} target`}
                                type="number"
                                step="0.01"
                                min={joint.lower ?? -Math.PI}
                                max={joint.upper ?? Math.PI}
                                value={
                                  Number.isFinite(goals[i])
                                    ? Number(goals[i].toFixed(3))
                                    : 0
                                }
                                disabled={disabled}
                                onChange={(e) => {
                                  const value = Number(e.target.value);
                                  setGoals((old) =>
                                    old.map((v, j) => (j === i ? value : v)),
                                  );
                                  setGoalDirty(true);
                                }}
                              />
                              <small>
                                {joint.kind === "prismatic" ? "m" : "rad"}
                              </small>
                            </div>
                          </div>
                          <input
                            id={`joint-${i}`}
                            aria-label={`${joint.name} joint goal`}
                            type="range"
                            min={joint.lower ?? -Math.PI}
                            max={joint.upper ?? Math.PI}
                            step="0.005"
                            value={goals[i] ?? 0}
                            disabled={disabled}
                            onChange={(e) => {
                              setGoals((old) =>
                                old.map((v, j) =>
                                  j === i ? Number(e.target.value) : v,
                                ),
                              );
                              setGoalDirty(true);
                            }}
                          />
                          <div className="joint-foot">
                            <span>
                              {joint.lower === null
                                ? "continuous"
                                : joint.lower.toFixed(1)}
                            </span>
                            <span>
                              actual <b>{nice(state?.q[i] ?? 0, 3)}</b>
                            </span>
                            <span>
                              {joint.upper?.toFixed(1) ?? "continuous"}
                            </span>
                          </div>
                        </div>
                      ) : null,
                    )}
                  </div>
                  <div className="goal-buttons">
                    <button
                      className="button apply-button"
                      disabled={disabled || !goals.length}
                      onClick={() =>
                        void command("joints", {
                          start: true,
                          arm: activeArm?.name,
                          values:
                            activeArm?.joint_indices.map((i) => goals[i]) ??
                            goals,
                        })
                      }
                    >
                      {changed ? (
                        <Circle size={13} fill="currentColor" />
                      ) : (
                        <Check size={14} />
                      )}
                      Move arm
                    </button>
                    <button
                      className="icon-button home-button"
                      aria-label="Set all arms to home joint goals"
                      title="Set all arms to home goals"
                      disabled={disabled}
                      onClick={() => void command("home", { start: true })}
                    >
                      <Home size={17} />
                    </button>
                  </div>
                  <p className="control-note">
                    Move arm plans and starts torque-controlled motion. Use
                    Pause motion to suspend execution.
                  </p>
                  <div className="target-control">
                    <div className="section-heading">
                      <span>CARTESIAN TARGET</span>
                      <Crosshair size={14} />
                    </div>
                    <div className="xyz-fields">
                      {(["X", "Y", "Z"] as const).map((axis, i) => (
                        <label key={axis}>
                          <span>{axis}</span>
                          <input
                            type="number"
                            aria-label={`Target ${axis}`}
                            step="0.01"
                            value={Number(target[i].toFixed(3))}
                            disabled={disabled}
                            onChange={(e) =>
                              setTarget(
                                (old) =>
                                  old.map((v, j) =>
                                    j === i ? Number(e.target.value) : v,
                                  ) as Vec3,
                              )
                            }
                          />
                          <small>m</small>
                        </label>
                      ))}
                    </div>
                    <button
                      className="button ik-button"
                      disabled={disabled}
                      onClick={() =>
                        void command("target", {
                          start: true,
                          arm: activeArm?.name,
                          values: target,
                        })
                      }
                    >
                      <Crosshair size={14} />
                      Move to target
                      <ArrowRight size={14} />
                    </button>
                    <div className="error-chart">
                      <span>RECENT TARGET ERROR</span>
                      <strong>
                        {state ? nice(state.distance * 1000, 2) : "—"}{" "}
                        <small>mm</small>
                      </strong>
                      <Sparkline data={history.map((s) => s.state.distance)} />
                    </div>
                  </div>
                </>
              }
            </>
          ) : (
            <>
              <div className="inspector-title">
                <div className="section-kicker">OBJECT INSPECTOR</div>
                <h2>
                  {selected === null
                    ? "Ground plane"
                    : model?.names[selected].replaceAll("_", " ")}
                </h2>
                <p>
                  {selected === null
                    ? "World reference · z = 0"
                    : `Link ${selected} · ${model?.kinds[selected]} joint`}
                </p>
              </div>
              <div className="inspect-symbol">
                {selected === null ? (
                  <Grid2X2 size={42} strokeWidth={1} />
                ) : (
                  <Box size={42} strokeWidth={1} />
                )}
              </div>
              <dl className="property-list">
                <div>
                  <dt>Type</dt>
                  <dd>
                    {selected === null
                      ? "Static plane"
                      : model?.kinds[selected]}
                  </dd>
                </div>
                {selected !== null && (
                  <>
                    <div>
                      <dt>Mass</dt>
                      <dd>{nice(model?.masses[selected] ?? 0)} kg</dd>
                    </div>
                    <div>
                      <dt>Parent</dt>
                      <dd>
                        {model && model.parents[selected] >= 0
                          ? model.names[model.parents[selected]]
                          : "World"}
                      </dd>
                    </div>
                    <div>
                      <dt>Position X</dt>
                      <dd>{nice(state?.positions[selected]?.[0] ?? 0)} m</dd>
                    </div>
                    <div>
                      <dt>Position Y</dt>
                      <dd>{nice(state?.positions[selected]?.[1] ?? 0)} m</dd>
                    </div>
                    <div>
                      <dt>Position Z</dt>
                      <dd>{nice(state?.positions[selected]?.[2] ?? 0)} m</dd>
                    </div>
                  </>
                )}
                {selectedJoint && (
                  <>
                    <div>
                      <dt>Joint position</dt>
                      <dd>
                        {nice(state?.q[selectedJointIndex] ?? 0)}{" "}
                        {selectedJoint.kind === "prismatic" ? "m" : "rad"}
                      </dd>
                    </div>
                    <div>
                      <dt>Joint velocity</dt>
                      <dd>
                        {nice(state?.qd[selectedJointIndex] ?? 0)}{" "}
                        {selectedJoint.kind === "prismatic" ? "m/s" : "rad/s"}
                      </dd>
                    </div>
                    <div>
                      <dt>Applied effort</dt>
                      <dd>
                        {nice(state?.torque[selectedJointIndex] ?? 0)}{" "}
                        {selectedJoint.kind === "prismatic" ? "N" : "N·m"}
                      </dd>
                    </div>
                    <div>
                      <dt>Effort limit</dt>
                      <dd>
                        {selectedJoint.effort}{" "}
                        {selectedJoint.kind === "prismatic" ? "N" : "N·m"}
                      </dd>
                    </div>
                  </>
                )}
              </dl>
              <button
                className="button inspect-to-control"
                onClick={() => setInspector("control")}
              >
                <SlidersHorizontal size={14} />
                Open controller
                <ArrowRight size={14} />
              </button>
              <p className="info-note">
                Select a robot link in the viewport or scene tree to inspect its
                live physical state.
              </p>
            </>
          )}
        </div>
        <div className="inspector-footer">
          <span className="tiny-dot" />
          <span>
            {replay !== null
              ? "Showing recorded state"
              : "Values from Python simulation"}
          </span>
        </div>
      </aside>{" "}
    </>
  );
}
