import {
  Box,
  ChevronRight,
  CircleHelp,
  FolderOpen,
  Layers3,
  Save,
} from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";

export function StudioHeader() {
  const {
    connection,
    projectName,
    setProjectName,
    setHelp,
    treeOpen,
    setTreeOpen,
    connected,
    disabled,
    save,
    model,
  } = useWorkspace();
  return (
    <>
      {" "}
      <header className="topbar">
        <a className="brand" href="/" aria-label="robogogogo Studio home">
          <span className="brand-symbol">
            <Box size={21} strokeWidth={1.7} />
          </span>
          <strong>
            robogogogo<span className="brand-dot">.</span>
          </strong>
          <span className="studio-tag">ROBOTICS STUDIO</span>
        </a>
        <div className="project-heading">
          <span className="divider" />
          <FolderOpen size={15} />
          <input
            aria-label="Experiment name"
            value={projectName}
            maxLength={80}
            onChange={(e) => setProjectName(e.target.value)}
          />
          <span className="local-tag">LOCAL</span>
        </div>
        <div className="header-actions">
          <a href="/">Workflow</a>
          <a href="/hardware/">Hardware</a>
          <a href="/structure/">Structural Lab</a>
          <button
            className="icon-button help-button"
            aria-label="Help and shortcuts"
            onClick={() => setHelp(true)}
          >
            <CircleHelp size={18} />
          </button>
          <button
            className="button save-button"
            disabled={disabled || !projectName.trim()}
            onClick={() => void save()}
          >
            <Save size={15} />
            Save experiment
          </button>
        </div>
      </header>
      <div className="workspace-toolbar">
        <div className="workspace-label">
          <button
            className="icon-button tree-toggle"
            onClick={() => setTreeOpen(!treeOpen)}
            aria-label="Toggle scene sidebar"
          >
            <Layers3 size={17} />
          </button>
          <span className="workspace-icon">
            <Layers3 size={15} />
          </span>
          <span>robotics workspace</span>
          <ChevronRight size={14} />
          <span className="subtle">{model?.name ?? "Connecting…"}</span>
        </div>
        <div className={`connection ${connected ? "online" : ""}`}>
          <i />
          {connected
            ? "Python engine connected"
            : connection === "connecting"
              ? "Connecting to Python engine"
              : "Engine disconnected · reconnecting"}
        </div>
      </div>{" "}
    </>
  );
}
