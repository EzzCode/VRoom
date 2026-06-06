export const CAPTURE_CONFIG = {
  /** Process every Nth frame in the worklet to save CPU/battery */
  frameSamplingInterval: 10,

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
    minDistance: 0.025, // Extremely dense: every 2.5cm
    /** Max cosine similarity (above this = "same direction"). 0.996 ≈ 5° */
    maxSimilarity: 0.996, // Extremely dense: every 5 degrees
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
    frustumDepth: 2.0,
  },
} as const;

export type CaptureConfig = typeof CAPTURE_CONFIG;
