import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { request, useSimulation } from "../api";
import type { ViewportHandle } from "../Viewport";
import { useVla } from "../vlaClient";
import type { ProjectSummary, Snapshot, Vec3 } from "../types";

function useWorkspaceState() {
  const { snapshot, history, connection, accept } = useSimulation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [inspector, setInspector] = useState<"task" | "control" | "inspect">(
    "task",
  );
  const [leftTab, setLeftTab] = useState<"scene" | "projects">("scene");
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [projectName, setProjectName] = useState("Untitled experiment");
  const [goals, setGoals] = useState<number[]>([]);
  const [target, setTarget] = useState<Vec3>([0.6, 0.15, 0.5]);
  const [goalDirty, setGoalDirty] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(false);
  const [help, setHelp] = useState(false);
  const [treeOpen, setTreeOpen] = useState(false);
  const [replay, setReplay] = useState<number | null>(null);
  const upload = useRef<HTMLInputElement>(null);
  const viewport = useRef<ViewportHandle>(null);
  const previousEpoch = useRef<number | null>(null);
  const vla = useVla({ viewport, accept, setBusy, setError, setNotice });
  const connected = connection === "connected";
  const disabled = !connected || busy || replay !== null;
  const shown =
    replay === null
      ? snapshot
      : (history[Math.min(replay, history.length - 1)] ?? snapshot);
  const loadProjects = useCallback(async () => {
    try {
      setProjects(await request<ProjectSummary[]>("/projects"));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    if (connected) {
      void loadProjects();
      void vla.refresh();
    }
  }, [connected, loadProjects]);
  useEffect(() => {
    if (!snapshot) return;
    if (previousEpoch.current !== snapshot.epoch) {
      previousEpoch.current = snapshot.epoch;
      setReplay(null);
      setSelected(null);
      setGoalDirty(false);
      setGoals(snapshot.state.goal);
      setTarget(snapshot.state.target);
    }
  }, [snapshot]);
  useEffect(() => {
    if (snapshot?.model.scene === "mobile_dual_arm") setInspector("control");
  }, [snapshot?.model.scene]);
  useEffect(() => {
    if (snapshot) setProjectName(snapshot.project.name);
  }, [snapshot?.project.id, snapshot?.project.name]);

  const perform = useCallback(
    async (action: () => Promise<Snapshot>, success?: string) => {
      setBusy(true);
      setError("");
      setNotice("");
      try {
        const next = await action();
        accept(next);
        setGoals(next.state.goal);
        setTarget(next.state.target);
        setGoalDirty(false);
        if (success) setNotice(success);
        return next;
      } catch (e) {
        setError((e as Error).message);
        return null;
      } finally {
        setBusy(false);
      }
    },
    [accept],
  );
  const command = useCallback(
    (op: string, extras: Record<string, unknown> = {}) =>
      perform(() => request<Snapshot>("/command", { op, ...extras })),
    [perform],
  );

  const save = useCallback(async () => {
    if (!snapshot || !projectName.trim()) return;
    await perform(async () => {
      const result = await request<{ snapshot: Snapshot }>("/projects", {
        name: projectName.trim(),
        id: snapshot.project.id,
      });
      return result.snapshot;
    }, "Experiment checkpoint saved on this computer.");
    await loadProjects();
  }, [snapshot, projectName, perform, loadProjects]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setHelp(false);
        setTreeOpen(false);
      }
      const editing = (event.target as HTMLElement)?.closest(
        'input, textarea, select, [contenteditable="true"]',
      );
      if ((event.metaKey || event.ctrlKey) && event.key === "s") {
        event.preventDefault();
        if (!disabled) void save();
        return;
      }
      if (editing || help || disabled || !snapshot) return;
      if (event.code === "Space") {
        event.preventDefault();
        void vla.capture();
      }
      if (event.key.toLowerCase() === "f")
        viewport.current?.view("perspective");
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [command, disabled, help, save, snapshot]);

  const importFile = async (file: File | undefined) => {
    if (!file) return;
    if (file.size > 262144) {
      setError(
        "URDF must be smaller than 256 KB. Mesh assets are not supported yet.",
      );
      return;
    }
    await perform(async () =>
      request<Snapshot>("/robot", { xml: await file.text() }),
    );
    if (upload.current) upload.current.value = "";
  };
  const chooseBody = (body: number) => {
    setSelected(body);
    setInspector("inspect");
  };
  const scrub = async (value: number) => {
    if (!snapshot || busy) return;
    if (snapshot.state.running) await command("pause");
    setReplay(value);
  };
  const state = shown?.state;
  const model = snapshot?.model;
  const selectedJoint = model?.joints.find((j) => j.body === selected);
  const selectedJointIndex =
    model?.joints.findIndex((j) => j.body === selected) ?? -1;
  const changed =
    goalDirty &&
    snapshot &&
    goals.some((v, i) => Math.abs(v - snapshot.state.goal[i]) > 0.00001);
  const trail = history
    .slice(0, replay === null ? undefined : replay + 1)
    .map(
      (s) =>
        s.state.positions[
          s.model.arms[0]?.body ?? s.state.positions.length - 1
        ],
    );
  const status = !connected
    ? "OFFLINE"
    : replay !== null
      ? "REPLAY"
      : snapshot?.state.running
        ? "RUNNING"
        : "PAUSED";

  return {
    snapshot,
    history,
    connection,
    busy,
    error,
    setError,
    notice,
    selected,
    setSelected,
    inspector,
    setInspector,
    leftTab,
    setLeftTab,
    projects,
    projectName,
    setProjectName,
    goals,
    setGoals,
    target,
    setTarget,
    setGoalDirty,
    showGrid,
    setShowGrid,
    showAxes,
    setShowAxes,
    help,
    setHelp,
    treeOpen,
    setTreeOpen,
    replay,
    setReplay,
    upload,
    viewport,
    vla,
    connected,
    disabled,
    shown,
    loadProjects,
    perform,
    command,
    save,
    importFile,
    chooseBody,
    scrub,
    state,
    model,
    selectedJoint,
    selectedJointIndex,
    changed,
    trail,
    status,
  };
}

const WorkspaceContext = createContext<ReturnType<
  typeof useWorkspaceState
> | null>(null);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const value = useWorkspaceState();
  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace() {
  const workspace = useContext(WorkspaceContext);
  if (!workspace)
    throw new Error("Workspace components require WorkspaceProvider");
  return workspace;
}
