// ────────────────────────────────────────────────────────────
// VRoom Shared Types
// ────────────────────────────────────────────────────────────

/** 3D vector (position, direction, etc.) */
export type Vec3 = [number, number, number];

/** Camera pose captured from ARKit/ARCore via ViroReact */
export interface CameraPose {
  /** Camera position in world space */
  position: Vec3;
  /** Camera rotation in Euler angles (degrees) */
  rotation: Vec3;
  /** Normalised forward direction vector */
  forward: Vec3;
  /** Normalised up direction vector */
  up: Vec3;
  /** Timestamp (ms since epoch) */
  timestamp: number;
}

/** A saved keyframe: image path + associated metadata */
export interface Keyframe {
  /** Absolute path to the saved JPEG on device */
  imagePath: string;
  /** Camera pose at capture time */
  pose: CameraPose;
  /** Laplacian variance (sharpness score) */
  blurScore: number;
  /** Frame index within the current session */
  index: number;
}

/** Result returned by a capture gate's evaluate() */
export interface GateResult {
  /** Whether the gate passed (true = allow capture) */
  passed: boolean;
  /** Human-readable guidance when the gate fails */
  reason?: string;
}

/** Voxel key for coverage tracking, "x_y_z" */
export type VoxelKey = string;

/** Tracking data for a single voxel */
export interface VoxelData {
  /** How many different cameras have observed this voxel */
  observationCount: number;
  /** Normalised viewing direction vectors from each observation */
  viewingDirs: Vec3[];
  /** Mean angular diversity across observations (radians) */
  angularDiversity: number;
}

/** Capture session metadata exported alongside images */
export interface SessionMetadata {
  /** Session start time (ISO 8601) */
  startedAt: string;
  /** Session end time (ISO 8601) */
  endedAt?: string;
  /** All saved keyframes */
  keyframes: Keyframe[];
  /** Coverage percentage at session end */
  coveragePercent: number;
  /** Total frames analysed */
  totalFramesAnalysed: number;
}
