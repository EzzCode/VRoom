export const CAPTURE_CONFIG = {
  /** Process every Nth frame in the worklet to save CPU/battery */
  frameSamplingInterval: 15,

  /** Resolution for the downscaled frame sent to OpenCV */
  resize: {
    width: 480,
    height: 640,
  },

  /** Laplacian variance threshold — below this is "blurry" */
  blurThreshold: 150,

  /** Top offset for the HUD overlay (accounts for status bar) */
  hudTopOffset: 60,

  /** Angle diversity gate thresholds */
  angleDiversity: {
    /** Minimum Euclidean distance (metres) to allow a new capture */
    minDistance: 0.10,
    /** Max cosine similarity (above this = "same direction") */
    maxSimilarity: 0.95,
  },

  /** Coverage voxel grid configuration (Build 3) */
  coverage: {
    /** Voxel cube size in metres */
    voxelSize: 0.15,
    /** Minimum observation count for a voxel to be "covered" */
    minObservations: 3,
    /** Camera field of view in degrees (used for frustum) */
    cameraFovDeg: 60,
    /** Frustum ray-cast depth in metres */
    frustumDepth: 5.0,
  },
} as const;

export type CaptureConfig = typeof CAPTURE_CONFIG;
