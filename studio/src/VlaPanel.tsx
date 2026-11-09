import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  Camera,
  Check,
  Code2,
  MessageSquare,
  Play,
  Plus,
  Square,
  StepForward,
} from "lucide-react";
import {
  PerceptionPanel,
  type StereoReport,
} from "./components/PerceptionPanel";
import { CoordinationPanel } from "./components/CoordinationPanel";
import type { Coordination } from "./types";
import { CameraMountEditor } from "./components/CameraMountEditor";
import type { CameraConfig, RobotModel, EpisodeSummary } from "./types";

interface Props {
  materials: Record<string, import("./types").MaterialParameters>;
  coordination: Coordination | null;
  taskRunning: boolean;
  onPlanTask: (
    kind: "inspection" | "handover" | "material_grasp",
    grip: import("./types").GripParameters,
  ) => void;
  manipulation: import("./types").ManipulationState | null;
  onRunTask: () => void;
  episode: EpisodeSummary | null;
  model: RobotModel | null;
  disabled: boolean;
  collecting: boolean;
  onStart: (
    instruction: string,
    variant: string,
    cameras: CameraConfig[],
  ) => void;
  onCapture: () => void;
  onStereo: (left: string, right: string) => void;
  stereo: StereoReport | null;
  onAct: (delta: number[]) => void;
  onCollect: () => void;
  onStop: () => void;
  onSave: () => void;
  onError: (message: string) => void;
}
export function VlaPanel(props: Props) {
  const episode = props.episode;
  const [instruction, setInstruction] = useState(
    "Move the robot tool to the highlighted target.",
  );
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const [variant, setVariant] = useState("a");
  const [action, setAction] = useState("[0.05, 0, 0, 0, 0, 0]");
  useEffect(() => {
    if (episode) {
      setInstruction(episode.instruction);
      setCameras(episode.cameras);
      setAction(
        JSON.stringify(
          episode.action_spec.names.map((_, i) => (i === 0 ? 0.05 : 0)),
        ),
      );
    }
  }, [episode?.id]);
  const execute = () => {
    try {
      const values: unknown = JSON.parse(action);
      if (
        !Array.isArray(values) ||
        values.length !== episode?.action_spec.dimensions ||
        !values.every((v) => typeof v === "number" && Number.isFinite(v))
      )
        throw new Error(
          `Enter an array of ${episode?.action_spec.dimensions ?? 6} finite joint deltas.`,
        );
      props.onAct(values);
    } catch (error) {
      props.onError((error as Error).message);
    }
  };
  return (
    <div className="vla-panel">
      <div className="inspector-title">
        <div className="section-kicker">PERCEPTION · PLANNING · CONTROL</div>
        <h2>Episode workspace</h2>
        <p>Observe. Act. Collect the next transition.</p>
      </div>
      <CoordinationPanel
        task={props.coordination}
        manipulation={props.manipulation}
        materials={props.materials}
        armCount={props.model?.arms.length ?? 0}
        disabled={props.disabled}
        running={props.taskRunning}
        onPlan={props.onPlanTask}
        onRun={props.onRunTask}
        onPause={props.onStop}
      />
      <div className="vla-instruction">
        <label htmlFor="instruction">
          <MessageSquare size={13} />
          TASK DESCRIPTION
        </label>
        <textarea
          id="instruction"
          value={instruction}
          maxLength={1000}
          disabled={props.disabled}
          onChange={(e) => setInstruction(e.target.value)}
          rows={3}
        />
        <div className="task-variant">
          <span>Task fixture</span>
          <select
            aria-label="Reach task variant"
            value={variant}
            disabled={props.disabled}
            onChange={(e) => setVariant(e.target.value)}
          >
            <option value="a">Reach target A</option>
            <option value="b">Reach target B</option>
            <option value="c">Reach target C</option>
          </select>
        </div>
        <CameraMountEditor
          cameras={cameras}
          model={props.model}
          disabled={props.disabled}
          onChange={setCameras}
        />
        <button
          className="button new-episode"
          disabled={props.disabled || !instruction.trim() || !cameras.length}
          onClick={() => props.onStart(instruction, variant, cameras)}
        >
          <Plus size={14} />
          Start new episode
          <ArrowRight size={14} />
        </button>
        <p className="vla-caption">
          The fixture sets the physical target; the instruction is a policy
          input, not a parsed robot command.
        </p>
      </div>
      <PerceptionPanel
        episode={episode}
        disabled={props.disabled}
        analyze={props.onStereo}
        report={props.stereo}
      />
      <div className="episode-stats">
        <div>
          <span>OBSERVATIONS</span>
          <strong>{String(episode?.observations ?? 0).padStart(3, "0")}</strong>
        </div>
        <div>
          <span>ACTIONS</span>
          <strong>{String(episode?.actions ?? 0).padStart(3, "0")}</strong>
        </div>
        <div>
          <span>OUTCOME</span>
          <strong
            className={episode?.success ? "success-text" : "outcome-text"}
          >
            {episode?.success
              ? "Success"
              : episode?.done
                ? "Ended"
                : "In progress"}
          </strong>
        </div>
      </div>
      {episode?.done && (
        <div className="episode-result">
          {episode.success ? <Check size={13} /> : <Square size={11} />}
          <span>{episode.reason}</span>
        </div>
      )}
      <div className="vla-block">
        <div className="section-heading">
          <span>ARM GROUPS</span>
        </div>
        {episode?.arms.map((arm) => (
          <p key={arm.name} className="vla-caption">
            <strong>{arm.name}</strong> · {arm.tool}
            <br />
            Action indices: {arm.joint_indices.join(", ")}
          </p>
        ))}
        <p className="vla-caption">
          One action vector controls all arms. Success requires every tool to
          reach its target. Collision checking is enabled in coordinated tasks;
          standalone manual and reach actions remain unguarded.
        </p>
      </div>
      <div className="vla-block">
        <div className="section-heading">
          <span>OBSERVATION CONTRACT</span>
          <Camera size={14} />
        </div>
        <div className="contract-rows">
          {episode?.cameras.map((camera) => (
            <div key={camera.name}>
              <span className="contract-dot rgb" />
              <strong>{camera.name}</strong>
              <span>
                {camera.width} × {camera.height} · RGB
              </span>
            </div>
          ))}
          <div>
            <span className="contract-dot language" />
            <strong>instruction</strong>
            <span>UTF-8 text</span>
          </div>
          <div>
            <span className="contract-dot state" />
            <strong>proprioception</strong>
            <span>q + q̇ · {episode?.action_spec.dimensions ?? 6} joints</span>
          </div>
        </div>
        <button
          className="button capture-button"
          disabled={props.disabled || !episode}
          onClick={props.onCapture}
        >
          <Camera size={14} />
          Capture aligned observation
        </button>
      </div>
      <div className="vla-block">
        <div className="section-heading">
          <span>ACTION CONTRACT</span>
          <Code2 size={14} />
        </div>
        <div className="action-format">
          <code>joint_position_delta</code>
          <span>un-normalized</span>
        </div>
        <label className="action-label" htmlFor="action">
          Manual action · ordered by robot joints
        </label>
        <textarea
          id="action"
          aria-label="Joint delta action"
          rows={2}
          spellCheck={false}
          value={action}
          disabled={props.disabled}
          onChange={(e) => setAction(e.target.value)}
        />
        <div className="action-details">
          <span>{episode?.action_spec.dimensions ?? 6} values · rad / m</span>
          <span>
            {episode?.action_spec.default_control_ticks ?? 10} control ticks
          </span>
        </div>
        <button
          className="button execute-button"
          disabled={
            props.disabled || !episode || episode.done || !!props.coordination
          }
          onClick={execute}
        >
          <StepForward size={14} />
          Execute + capture next RGB
        </button>
      </div>
      <div className="reference-block">
        <div className="section-heading">
          <span>CONTROL SYSTEM</span>
          <span className="baseline-tag">RULE-BASED</span>
        </div>
        <p>
          Rule-based reach → action supervisor → joint controller. Uses
          simulator target positions. Learned policies share the same
          observation and action interface.
        </p>
        {props.collecting ? (
          <button className="button collect-button" onClick={props.onStop}>
            <Square size={12} fill="currentColor" />
            Stop after current action
          </button>
        ) : (
          <button
            className="button collect-button"
            disabled={
              props.disabled || !episode || episode.done || !!props.coordination
            }
            onClick={props.onCollect}
          >
            <Play size={12} fill="currentColor" />
            Run rule-based reach
          </button>
        )}
      </div>
      <div className="vla-export">
        <button
          className="button"
          disabled={props.disabled || !episode?.observations}
          onClick={props.onSave}
        >
          <ArrowDownToLine size={14} />
          Save & export episode
        </button>
        <p>ZIP · RGB + ground-truth depth + transitions + physics parameters</p>
      </div>
      <div className="vla-limit">
        Frictional finger contacts support the handover fixture. Model-specific
        inference and training adapters are the next integration layer.
      </div>
    </div>
  );
}
