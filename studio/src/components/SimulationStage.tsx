import { Viewport } from "../Viewport";
import {
  Activity,
  Box,
  Camera,
  ArrowDownToLine,
  ArrowRight,
  Axis3D,
  Clock3,
  Grid2X2,
  LoaderCircle,
  Maximize2,
  MousePointer2,
  Radio,
  RotateCcw,
  Target,
} from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";
import { nice, formatTime } from "../format";

export function SimulationStage() {
  const {
    snapshot,
    history,
    busy,
    selected,
    posePreview,
    goalDirty,
    showGrid,
    setShowGrid,
    showAxes,
    setShowAxes,
    showInternals,
    setShowInternals,
    showCameraGuides,
    setShowCameraGuides,
    replay,
    setReplay,
    viewport,
    vla,
    connected,
    disabled,
    shown,
    chooseBody,
    scrub,
    state,
    model,
    trail,
    status,
  } = useWorkspace();
  return (
    <>
      {" "}
      <section className="stage-column">
        <div className="simulation-bar">
          <div className="transport">
            <button
              className="button primary"
              disabled={disabled || !snapshot?.episode}
              onClick={() => void vla.capture()}
            >
              <Camera size={15} />
              Capture observation<kbd>Space</kbd>
            </button>
            <button
              className="icon-button"
              title="New episode with the current instruction"
              aria-label="Reset episode"
              disabled={disabled || !snapshot?.episode}
              onClick={() =>
                void vla.start(
                  snapshot?.episode?.instruction ??
                    "Move the tool to the highlighted target.",
                  "a",
                )
              }
            >
              <RotateCcw size={16} />
            </button>
          </div>
          <div className="timing">
            <span className="timing-label">CONTROL ACTION</span>
            <strong>
              {Math.round(
                (snapshot?.episode?.action_spec.default_control_ticks ?? 10) *
                  (snapshot?.model.dt ?? 0.002) *
                  (snapshot?.model.substeps ?? 5) *
                  1000,
              )}
              <span> ms</span>
            </strong>
            <span className="small-divider" />
            <span>RGB + state</span>
          </div>
        </div>
        <div className="viewport-wrap">
          <div
            className="robot-view-switch"
            role="group"
            aria-label="Robot visualization mode"
          >
            <button
              aria-pressed={!showInternals}
              onClick={() => setShowInternals(false)}
            >
              Exterior
            </button>
            <button
              aria-pressed={showInternals}
              onClick={() => setShowInternals(true)}
            >
              Engineering
            </button>
          </div>
          {showInternals && (
            <div className="engineering-note">
              3 motors per shoulder · Mechanical inspection. Electrical
              packaging deferred; covers are removable concepts.
            </div>
          )}
          {shown ? (
            <Viewport
              ref={viewport}
              snapshot={shown}
              trail={trail}
              selected={selected}
              onSelect={chooseBody}
              posePreview={replay === null && goalDirty ? posePreview : null}
              showGrid={showGrid}
              showAxes={showAxes}
              showCameraGuides={showCameraGuides}
              showInternals={showInternals}
            />
          ) : (
            <div className="loading-scene">
              <span className="loading-logo">
                <Box size={30} />
              </span>
              <h2>Preparing your workspace</h2>
              <p>Connecting to the Python simulation engine.</p>
              <LoaderCircle className="spin" size={20} />
            </div>
          )}
          <div className="viewport-top">
            <div className="view-label">
              <span className={`status-dot ${status.toLowerCase()}`} />
              {status}
              <span className="small-divider" />
              <span className="view-subtitle">
                {replay !== null
                  ? "Recorded state"
                  : goalDirty
                    ? "POSE PREVIEW · outline only"
                    : "Perspective viewport"}
              </span>
            </div>
            <div className="view-badge">
              <span>Z</span>UP <span>·</span> METERS
            </div>
          </div>
          <div className="viewport-tools">
            <button
              className={`icon-button ${showGrid ? "active" : ""}`}
              aria-label="Toggle ground grid"
              aria-pressed={showGrid}
              onClick={() => setShowGrid(!showGrid)}
              title="Ground grid"
            >
              <Grid2X2 size={17} />
            </button>
            <button
              className={`icon-button ${showAxes ? "active" : ""}`}
              aria-label="Toggle coordinate frames"
              aria-pressed={showAxes}
              onClick={() => setShowAxes(!showAxes)}
              title="Coordinate frames"
            >
              <Axis3D size={18} />
            </button>

            <button
              className={`icon-button ${showCameraGuides ? "active" : ""}`}
              aria-label="Toggle camera guidelines"
              aria-pressed={showCameraGuides}
              title="Camera field-of-view guidelines"
              onClick={() => setShowCameraGuides(!showCameraGuides)}
            >
              <Camera size={18} />
            </button>
            <span />
            <button
              className="icon-button"
              aria-label="Frame robot"
              onClick={() => viewport.current?.view("perspective")}
              title="Frame robot (F)"
            >
              <Maximize2 size={17} />
            </button>
            <button
              className="view-button"
              onClick={() => viewport.current?.view("top")}
            >
              TOP
            </button>
            <button
              className="view-button"
              onClick={() => viewport.current?.view("front")}
            >
              FRONT
            </button>
          </div>
          <div className="scene-caption">
            <span className="scene-number">01</span>
            <div>
              <strong>{model?.name ?? "Robotics workspace"}</strong>
              <span>RG Mobile Duo · coordinated arm and torso motion</span>
            </div>
          </div>
          <div className="viewport-hint vla-hidden">
            <MousePointer2 size={12} />
            <span>Drag to orbit</span>
            <span>·</span>
            <span>Scroll to zoom</span>
          </div>
          {busy && (
            <div className="busy-badge">
              <LoaderCircle size={14} className="spin" />
              Applying changes…
            </div>
          )}
        </div>
        <div className="metrics">
          <div>
            <span>
              <Clock3 size={12} />
              SIMULATION TIME
            </span>
            <strong>
              {state ? formatTime(state.time) : "00.00"}
              <small>s</small>
            </strong>
          </div>
          <div>
            <span>
              <Target size={12} />
              MAX TOOL ERROR
            </span>
            <strong>
              {state ? nice(state.distance * 1000, 2) : "—"}
              <small>mm</small>
            </strong>
          </div>
          <div>
            <span>
              <Activity size={12} />
              MECHANICAL ENERGY
            </span>
            <strong>
              {state ? nice(state.energy, 2) : "—"}
              <small>J</small>
            </strong>
          </div>
          <div>
            <span>
              <Radio size={12} />
              REAL-TIME FACTOR
            </span>
            <strong>
              {state?.running ? nice(state.rtf, 2) : "—"}
              <small>×</small>
            </strong>
          </div>
        </div>
        <div className="timeline">
          <div className="timeline-heading">
            <div>
              <span className="record-dot" />
              <strong>Episode dynamics</strong>
              <span className="recording-label">
                {snapshot?.state.samples ?? 0} physics samples
              </span>
            </div>
            <div>
              {replay !== null && (
                <button className="live-button" onClick={() => setReplay(null)}>
                  Return to live
                  <ArrowRight size={12} />
                </button>
              )}
              <a
                className={`export-link ${!connected ? "unavailable" : ""}`}
                href="/api/export"
                download
                aria-disabled={!connected}
              >
                <ArrowDownToLine size={13} />
                Export .npz
              </a>
            </div>
          </div>
          <div className="timeline-track">
            <span>{history[0]?.state.time.toFixed(2) ?? "0.00"} s</span>
            <div className="timeline-slider">
              <div className="tick-marks" />
              <input
                aria-label="Scrub recorded trajectory"
                type="range"
                min={0}
                max={Math.max(history.length - 1, 1)}
                value={replay ?? Math.max(history.length - 1, 0)}
                disabled={busy || history.length < 2 || !connected}
                onChange={(e) => void scrub(Number(e.target.value))}
              />
            </div>
            <span>{history.at(-1)?.state.time.toFixed(2) ?? "0.00"} s</span>
          </div>
          <div className="timeline-foot">
            <span>
              {replay !== null
                ? "Inspecting a recorded browser frame. Physics is paused."
                : "Inspect recent motion here. Save a robot episode for aligned RGB, instruction, and actions."}
            </span>
            <span>
              {snapshot
                ? `${Math.round((snapshot.state.samples / snapshot.model.max_samples) * 100)}% buffer`
                : ""}
            </span>
          </div>
        </div>
      </section>{" "}
    </>
  );
}
