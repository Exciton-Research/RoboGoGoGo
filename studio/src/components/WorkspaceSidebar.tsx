import {
  Activity,
  Database,
  ArrowDownToLine,
  ArrowRight,
  Box,
  ChevronDown,
  ChevronRight,
  Circle,
  Code2,
  Crosshair,
  FileBox,
  FolderOpen,
  Grid2X2,
  Layers3,
  Plus,
  Save,
  Square,
} from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";
import { request } from "../api";
import type { Snapshot } from "../types";

export function WorkspaceSidebar() {
  const {
    snapshot,
    selected,
    setSelected,
    setInspector,
    leftTab,
    setLeftTab,
    projects,
    treeOpen,
    upload,
    vla,
    disabled,
    loadProjects,
    perform,
    save,
    importFile,
    chooseBody,
    command,
    model,
  } = useWorkspace();
  return (
    <>
      {" "}
      <aside className={`sidebar ${treeOpen ? "mobile-open" : ""}`}>
        <div className="panel-tabs">
          <button
            className={leftTab === "scene" ? "active" : ""}
            onClick={() => setLeftTab("scene")}
          >
            <Layers3 size={14} />
            Workspace
          </button>
          <button
            className={leftTab === "projects" ? "active" : ""}
            onClick={() => {
              setLeftTab("projects");
              void loadProjects();
              void vla.refresh();
            }}
          >
            <Database size={14} />
            Data
          </button>
        </div>
        {leftTab === "scene" ? (
          <>
            <div className="scene-select">
              <label>ROBOT TASK ENVIRONMENT</label>
              <div className="task-scene-name">
                <Crosshair size={14} />
                <span>RG Mobile Duo workspace</span>
              </div>
              <p className="fixture-caption">
                {model?.arms.length ?? 1} arm(s) · {model?.dof ?? 6} joints
              </p>
            </div>
            <p className="fixture-caption">
              RG Mobile Duo · wheeled base, folding torso, two 7-joint arms
            </p>
            <div className="section-heading">
              <span>ROBOT & OBSERVATION</span>
              <span className="count">{(model?.names.length ?? 0) + 1}</span>
            </div>
            <div className="scene-tree">
              <div className="tree-world">
                <ChevronDown size={13} />
                <Box size={15} />
                <strong>World</strong>
                <span className="tree-fixed">fixed</span>
              </div>
              <button
                className={`tree-row ${selected === null ? "selected" : ""}`}
                onClick={() => {
                  setSelected(null);
                  setInspector("inspect");
                }}
              >
                <span className="tree-branch" />
                <Grid2X2 size={14} />
                <span>Ground plane</span>
                <span className="tree-tag">z = 0</span>
              </button>
              <div className="tree-world robot-root">
                <ChevronDown size={13} />
                <Activity size={15} />
                <strong>
                  {model?.scene === "drop" ? "Contact body" : "Robot"}
                </strong>
                <span className="tree-tag">{model?.dof ?? "—"} DOF</span>
              </div>
              {model?.names.map((name, i) => (
                <button
                  key={name}
                  className={`tree-row link-row ${selected === i ? "selected" : ""}`}
                  onClick={() => chooseBody(i)}
                  title={`Inspect ${name}`}
                >
                  <span className="tree-branch" />
                  {model.kinds[i] === "fixed" ? (
                    <Square size={12} />
                  ) : (
                    <Circle size={12} />
                  )}
                  <span>{name.replaceAll("_", " ")}</span>
                  <span className="joint-marker">
                    {model.kinds[i] === "fixed"
                      ? "F"
                      : model.kinds[i] === "prismatic"
                        ? "P"
                        : "R"}
                  </span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="projects-list">
            <div className="section-heading">
              <span>WORKSPACE CHECKPOINTS</span>
              <span className="count">{projects.length}</span>
            </div>
            {!projects.length ? (
              <div className="empty-projects">
                <FolderOpen size={28} />
                <h3>Your experiments live here</h3>
                <p>
                  Save a checkpoint to preserve the robot state and controller
                  targets.
                </p>
                <button
                  className="button"
                  onClick={() => void save()}
                  disabled={disabled}
                >
                  <Save size={14} />
                  Save first experiment
                </button>
              </div>
            ) : (
              projects.map((p) => (
                <button
                  className={`project-card ${snapshot?.project.id === p.id ? "current" : ""}`}
                  disabled={disabled}
                  key={p.id}
                  onClick={() =>
                    void perform(() =>
                      request<Snapshot>(`/projects/${p.id}/load`, {}),
                    )
                  }
                >
                  <FileBox size={18} />
                  <span>
                    <strong>{p.name}</strong>
                    <small>
                      {p.scene === "arm"
                        ? "6-axis arm"
                        : p.scene === "drop"
                          ? "Contact lab"
                          : "Imported robot"}{" "}
                      · {p.time.toFixed(2)} s
                    </small>
                    <small>
                      {new Date(p.updated).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </small>
                  </span>
                  <ChevronRight size={14} />
                </button>
              ))
            )}
            <div className="dataset-heading">
              <span>COLLECTED EPISODES</span>
              <span className="count">{vla.episodes.length}</span>
            </div>
            {!vla.episodes.length ? (
              <p className="dataset-empty">
                Capture observations and save an episode to begin a local
                dataset.
              </p>
            ) : (
              vla.episodes.map((episode) => (
                <a
                  className="episode-card"
                  key={episode.id}
                  href={`/api/vla/episodes/${episode.id}`}
                  download
                >
                  <Database size={16} />
                  <span>
                    <strong>{episode.instruction}</strong>
                    <small>
                      {episode.observations} RGB · {episode.actions} actions
                    </small>
                    <small>
                      {episode.success ? "Reach success" : "Partial episode"} ·
                      Download ZIP
                    </small>
                  </span>
                  <ArrowDownToLine size={13} />
                </a>
              ))
            )}
          </div>
        )}
        <div className="sidebar-bottom">
          <div className="engine-card">
            <span className="engine-icon">
              <Code2 size={17} />
            </span>
            <div>
              <strong>Native physics runtime</strong>
              <small>
                {model?.backend ?? "Python + JAX"} <span>·</span>{" "}
                {model?.dtype ?? "—"}
              </small>
            </div>
          </div>
          <p>
            Instruction → observation → action
            <br />
            Model-agnostic episode contract
          </p>
          <a href="/docs" target="_blank" rel="noreferrer">
            Explore the Python API
            <ArrowRight size={13} />
          </a>
        </div>
      </aside>{" "}
    </>
  );
}
