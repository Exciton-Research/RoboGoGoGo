export type TaskState = {
  object: LibraryObject;
  scene_id: string;
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
  revision: number;
  session_id: string;
  ended: boolean;
  episode_id: string;
  step_index: number;
  seed: number;
  mode: "manual" | "script" | "replay";
  joint_target: number[];
  cameras: Record<
    string,
    { position: number[]; quaternion: number[]; fovy: number }
  >;
  recording: { id: string; transitions: number; rgb: boolean } | null;
  episodes: EpisodeSummary[];
  last_saved: (EpisodeSummary & { observations: number }) | null;
  replay: {
    id: string;
    frame: number;
    frames: number;
    playing: boolean;
    rgb: boolean;
    outcome: string;
  } | null;
};

export type EpisodeSummary = {
  object_name?: string;
  id: string;
  created: string;
  seed: number;
  transitions: number;
  duration: number;
  outcome: string;
  rgb: boolean;
};

export type LibraryObject = {
  id: string;
  name: string;
  category: string;
  ready: boolean;
  dimensions?: number[];
  mass?: number;
  collision?: string;
  source: string | null;
  license?: string;
};

export type SceneData = {
  scene_id: string;
  objects: LibraryObject[];
  meshes: Record<
    string,
    { vertices: string; faces: string; uv?: string; normals?: string }
  >;
  geoms: {
    type: number;
    mesh: number;
    size: number[];
    rgba: number[];
    name: string | null;
    texture?: string | null;
  }[];
  source: string;
  legacy_workspace: boolean;
  state: TaskState;
};
