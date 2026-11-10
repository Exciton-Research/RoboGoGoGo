export type Vec3 = [number, number, number];
export interface Joint {
  name: string;
  body: number;
  kind: string;
  lower: number | null;
  upper: number | null;
  effort: number;
  home: number;
}
export interface Visual {
  body: number;
  shape: "box" | "sphere" | "cylinder";
  dimensions: number[];
  origin: Vec3;
  rpy: Vec3;
}
export interface RobotModel {
  name: string;
  scene: "arm" | "dual_arm" | "dual_arm_head" | "drop" | "urdf";
  dof: number;
  arms: ArmGroup[];
  camera_presets: Record<string, CameraConfig[]>;
  names: string[];
  parents: number[];
  kinds: string[];
  masses: number[];
  axes: Vec3[];
  joints: Joint[];
  dt: number;
  substeps: number;
  visuals: Visual[];
  spheres: { body: number; center: Vec3; radius: number }[];
  backend: string;
  dtype: string;
  max_samples: number;
}
export interface SimState {
  q: number[];
  qd: number[];
  goal: number[];
  target: Vec3;
  targets: Record<string, Vec3>;
  torque: number[];
  tool_positions: Record<string, Vec3>;
  target_errors: Record<string, number>;
  positions: Vec3[];
  rotations: number[][][];
  time: number;
  energy: number;
  distance: number;
  running: boolean;
  speed: number;
  samples: number;
  rtf: number;
  contacts: number;
  normal_force: number;
  penetration: number;
  message: string;
  error: string | null;
}
export interface EpisodeSummary {
  id: string;
  instruction: string;
  task: string;
  target: Vec3;
  targets: Record<string, Vec3>;
  observations: number;
  actions: number;
  horizon: number;
  done: boolean;
  success: boolean;
  reason: string | null;
  camera: CameraConfig;
  cameras: CameraConfig[];
  arms: ArmGroup[];
  action_spec: {
    arms: ArmGroup[];
    representation: string;
    dimensions: number;
    names: string[];
    units: string[];
    max_delta: number[];
    default_control_ticks: number;
    gripper: boolean;
  };
}
export interface DatasetEpisode {
  id: string;
  instruction: string;
  observations: number;
  actions: number;
  success: boolean;
  created: string;
}
export type ObjectMaterialName =
  "rigid_reference" | "soft_foam" | "soft_elastomer" | "fragile_shell";
export interface MaterialParameters {
  name: string;
  modulus_pa: number;
  thickness_m: number;
  relaxation_s: number;
  yield_strain: number;
  failure_strain: number;
  strength_pa: number;
  plastic_rate_s: number;
  mass_kg: number;
  initial_force_n: number;
  force_threshold_n: number;
  color: string;
}
export interface ObjectMaterialState {
  name: ObjectMaterialName;
  model: string;
  parameters?: MaterialParameters;
  indentation_m?: number[][];
  permanent_m?: number[][];
  damage?: number[][];
  maximum_indentation_m?: number;
  maximum_permanent_m?: number;
  damage_fraction?: number;
  damaged: boolean;
  fractured: boolean;
  peak_pressure_pa?: number;
  peak_strain?: number;
  risk?: number;
}
export interface GripParameters {
  active_perception: boolean;
  camera_recovery: boolean;
  recovery_behaviors: boolean;
  structural_dynamics: boolean;
  object_material: ObjectMaterialName;
  adaptive_grasp: boolean;
  initial_grip_force_n: number;
  object_damage_force_n: number;
  grip_force_n: number;
  friction_static: number;
  friction_dynamic: number;
}
export interface ManipulationState {
  recovery?: {
    phase: string;
    active: boolean;
    reason: string | null;
    primary: string | null;
    helper: string | null;
    support_index: number | null;
    attempts: number;
    support_verified_s: number;
    active_perception: {
      attempts: number;
      max_views: number;
      camera: string | null;
      capture_after: number | null;
      selection?: {
        gain: number;
        newly_visible: number;
        probes: number;
      } | null;
    } | null;
    camera_mode: boolean;
    perception: {
      source: string;
      grasp_count: number;
      support_count: number;
      uncertainty_m: number;
      diagnostics: {
        error?: string;
        scene_memory?: {
          surfaces: number;
          remembered: number;
          invalidated: number;
          motion_evidence: number;
          max_uncertainty_m: number;
        };
      };
    } | null;
    risk_hold: boolean;
    search_checks: number;
    route_waypoints: number;
    route_progress: number;
    selected_grasp: number[] | null;
    rejected_candidates: {
      reason: string;
      candidate?: string;
      support?: number;
    }[];
  } | null;
  support_load_n?: number[];
  motion_hold: boolean;
  adaptive_grasp: Record<
    string,
    {
      status: string;
      force_setpoint_n: number;
      filtered_slip_speed_m_s: number;
      peak_force_n: number;
      damage_force_n: number;
      fault: string | null;
      motion_hold: boolean;
    }
  >;
  object: {
    name: string;
    shape: "capsule";
    material: ObjectMaterialState;
    structure?: {
      positions: number[][][];
      triangles: number[][];
      damage: number[][];
      fragments: { cells: number[] }[];
      broken_bonds: number;
      bend_m: number;
    } | null;
    mass: number;
    radius: number;
    half_length: number;
    position: Vec3;
    rotation: number[][];
    velocity: Vec3;
    angular_velocity: Vec3;
    owner: string | null;
    damage_limit_exceeded: boolean;
  };
  grippers: Record<
    string,
    {
      width: number;
      command: "open" | "closed";
      holding: boolean;
      force_limit_n: number;
      mu_static: number;
      mu_dynamic: number;
      normal_force_n: number[];
      friction_force_n: number[];
      slip_speed_m_s: number;
      slipping: boolean;
      contact: boolean;
    }
  >;
  events: {
    event: string;
    arm?: string;
    giver?: string;
    receiver?: string;
    time: number;
  }[];
  grasp_model: string;
  payload_dynamics: boolean;
  contact_dt: number;
  maximum_penetration_m: number;
  failure: string | null;
}
export interface Coordination {
  kind: "inspection" | "handover" | "material_grasp";
  name: string;
  status: "ready" | "running" | "paused" | "blocked" | "completed";
  reason: string | null;
  stage_index: number;
  stage_count: number;
  stage_name: string;
  owner: string | null;
  waypoint: number;
  waypoint_count: number;
  minimum_clearance_m: number | null;
  margin_m: number;
  stages: {
    name: string;
    owner: string | null;
    method: string;
    waypoints: number;
  }[];
  events: {
    stage: number;
    name: string;
    owner: string | null;
    completed_at: number;
  }[];
  station: Vec3;
  obstacles: (
    | { kind: "sphere"; name: string; center: Vec3; radius: number }
    | { kind: "box"; name: string; center: Vec3; size: Vec3 }
  )[];
}
export interface Snapshot {
  object_materials: Record<string, MaterialParameters>;
  manipulation: ManipulationState | null;
  coordination: Coordination | null;
  sequence: number;
  epoch: number;
  model: RobotModel;
  project: { id: string | null; name: string };
  state: SimState;
  episode: EpisodeSummary | null;
}
export interface ProjectSummary {
  id: string;
  name: string;
  updated: string;
  scene: string;
  time: number;
}
export type Connection = "connecting" | "connected" | "disconnected";

export interface CameraConfig {
  name: string;
  width: number;
  height: number;
  mount?: string | null;
  body?: number | null;
  position: Vec3;
  look_at: Vec3;
  rpy?: Vec3 | null;
  up: Vec3;
  vertical_fov_degrees: number;
  near: number;
  far: number;
}

export interface ArmGroup {
  name: string;
  tool: string;
  body: number;
  joint_indices: number[];
  joint_names: string[];
}
