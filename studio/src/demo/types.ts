export type TaskState = {
  status: "ready" | "running" | "paused" | "complete" | "failed";
  message: string;
  phase: number;
  time: number;
  phases: string[];
  progress: number;
  poses: number[][];
  block: number[];
  tool: number[];
  hand_position: number[];
  hand_rotation: number[];
  target: number[];
  pick: number[];
  finger_gap_mm: number;
  lift_mm: number;
  placement_error_mm: number;
  finger_contacts: number;
};

export type SceneData = {
  meshes: Record<string, { vertices: string; faces: string }>;
  geoms: {
    type: number;
    mesh: number;
    size: number[];
    rgba: number[];
    name: string | null;
  }[];
  source: string;
  legacy_workspace: boolean;
  state: TaskState;
};
