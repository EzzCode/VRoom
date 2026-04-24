// ────────────────────────────────────────────────────────────
// VRoom Shared Types
// ────────────────────────────────────────────────────────────

/** 3D vector (position, direction, etc.) */
export type Vec3 = [number, number, number];

/** 4x4 matrix in row-major order */
export type Matrix4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
];

export type TrackingState = 'unavailable' | 'limited' | 'normal';

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  distortion: number[];
  source: string;
}

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
  /** Timestamp (nanoseconds since Unix epoch) */
  timestampNs: number;
  /** AR tracking quality for this pose */
  trackingState: TrackingState;
  /** Camera-to-world transform in meters, row-major */
  cameraToWorld: Matrix4;
}

/** A saved keyframe: image path + associated metadata */
export interface Keyframe {
  /** Stable frame ID shared across images, poses, and intrinsics */
  frameId: string;
  /** Absolute path to the saved JPEG on device */
  imagePath: string;
  /** Camera pose at capture time */
  pose: CameraPose;
  /** Captured image width in pixels */
  width: number;
  /** Captured image height in pixels */
  height: number;
  /** Intrinsics matched to the saved image */
  intrinsics?: CameraIntrinsics;
  /** Laplacian variance (sharpness score) */
  qualityScore: number;
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

/** Info about a 3D mesh file available for AR projection */
export interface MeshInfo {
  id: string;
  name: string;
  format: 'GLB' | 'OBJ';
  size: number;
  uri: string;
  thumbnailUri?: string;
  isBundled: boolean;
}

/** Capture session metadata exported alongside images */
export interface SessionMetadata {
  /** Session capture ID */
  captureId: string;
  /** Session start time (ISO 8601) */
  startedAt: string;
  /** Session end time (ISO 8601) */
  endedAt?: string;
  /** All saved keyframes */
  keyframes: Keyframe[];
  /** Overall capture status */
  captureStatus: 'completed' | 'interrupted' | 'aborted';
  /** Coverage percentage at session end */
  coveragePercent: number;
  /** Total frames analysed */
  totalFramesAnalysed: number;
}
