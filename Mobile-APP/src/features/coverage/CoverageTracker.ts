// ────────────────────────────────────────────────────────────
// CoverageTracker — voxel grid of observed space
// ────────────────────────────────────────────────────────────
//
// Pure module: no React, no Viro, no I/O. Given a CameraPose it marks
// which voxels the camera frustum covers and reports a coverage metric
// suitable for HUD display and the CoverageGate.
//
// Voxel state machine:
//   unseen   → not present in `voxels` map
//   partial  → in map, observationCount < minObservations
//   covered  → in map, observationCount >= minObservations
//
// "coveragePercent" is the ratio of covered voxels to all voxels we've
// ever touched. It represents the quality of what we have so far, not
// progress against an unknown total volume.
// ────────────────────────────────────────────────────────────

import { CameraPose, Vec3, VoxelKey } from '../../shared/core/types';
import { cross, normalize, subtract, dot, length as vecLength } from '../../shared/core/math';

export interface CoverageTrackerConfig {
  /** Edge length of one voxel cube, in metres */
  voxelSize: number;
  /** Observation count at which a voxel becomes "covered" */
  minObservations: number;
  /** Horizontal field of view in degrees */
  fovDeg: number;
  /** How far ahead of the camera to mark voxels, in metres */
  frustumDepth: number;
  /** Aspect ratio (width / height) for the sampled frustum. Default 4/3. */
  aspect?: number;
  /** Skip marking voxels closer than this (avoids "the wall in front of my lens") */
  nearPlane?: number;
}

interface VoxelEntry {
  /** Integer grid coordinates */
  ix: number;
  iy: number;
  iz: number;
  observationCount: number;
}

export interface ObserveResult {
  /** Voxels that just transitioned to covered on this call */
  newlyCovered: number;
  /** Voxels that were newly added (first ever observation) */
  newlyTouched: number;
  /** All voxel keys observed by this pose */
  observedKeys: VoxelKey[];
}

export interface VoxelView {
  key: VoxelKey;
  /** World-space centre of the voxel */
  center: Vec3;
  state: 'partial' | 'covered';
  observationCount: number;
}

export class CoverageTracker {
  private readonly config: Required<CoverageTrackerConfig>;
  private readonly voxels = new Map<VoxelKey, VoxelEntry>();
  private coveredCount = 0;

  constructor(config: CoverageTrackerConfig) {
    this.config = {
      aspect: 4 / 3,
      nearPlane: 0.2,
      ...config,
    };
  }

  /**
   * Mark voxels visible from this pose. Returns counts useful for the
   * CoverageGate (whether this frame added information) and HUD.
   *
   * Call this only after a frame has been accepted as a keyframe. For
   * gate evaluation (before capture) use `peek()` which is non-mutating.
   */
  observe(pose: CameraPose): ObserveResult {
    const keys = this.sampleFrustumVoxels(pose);
    let newlyCovered = 0;
    let newlyTouched = 0;
    const seenThisFrame = new Set<VoxelKey>();

    for (const { key, ix, iy, iz } of keys) {
      if (seenThisFrame.has(key)) continue;
      seenThisFrame.add(key);

      const { touched, covered } = this.markVoxel(ix, iy, iz, key);
      if (touched) newlyTouched += 1;
      if (covered) newlyCovered += 1;
    }

    return {
      newlyCovered,
      newlyTouched,
      observedKeys: Array.from(seenThisFrame),
    };
  }

  /**
   * Mark the single voxel that contains a world-space point — e.g. an AR
   * hit-test result on a real surface. Unlike observe(), this does NOT fill
   * the camera frustum volume, so voxels coat actual geometry instead of
   * empty air. Intended for hit-test-driven coverage (the demo).
   */
  observePoint(point: Vec3): ObserveResult {
    const s = this.config.voxelSize;
    const ix = Math.floor(point[0] / s);
    const iy = Math.floor(point[1] / s);
    const iz = Math.floor(point[2] / s);
    const key: VoxelKey = `${ix}_${iy}_${iz}`;
    const { touched, covered } = this.markVoxel(ix, iy, iz, key);
    return {
      newlyCovered: covered ? 1 : 0,
      newlyTouched: touched ? 1 : 0,
      observedKeys: [key],
    };
  }

  /**
   * Apply one observation to a single voxel, creating it if new and updating
   * the covered count when it crosses the threshold. Shared by observe() and
   * observePoint(). Returns whether the voxel was newly touched / newly covered.
   */
  private markVoxel(
    ix: number,
    iy: number,
    iz: number,
    key: VoxelKey,
  ): { touched: boolean; covered: boolean } {
    let entry = this.voxels.get(key);
    let touched = false;
    if (!entry) {
      entry = { ix, iy, iz, observationCount: 0 };
      this.voxels.set(key, entry);
      touched = true;
    }
    const before = entry.observationCount;
    entry.observationCount = before + 1;
    let covered = false;
    if (before < this.config.minObservations && entry.observationCount >= this.config.minObservations) {
      this.coveredCount += 1;
      covered = true;
    }
    return { touched, covered };
  }

  /**
   * Predict what `observe(pose)` would yield without mutating state.
   * Used by CoverageGate to decide pass/fail before the frame is saved.
   *
   * - `newlyTouched`: voxels never observed before (count 0 → 1).
   * - `advancesPartial`: already-touched voxels still below the covered
   *   threshold that this pose would push one observation closer (includes
   *   the ones that would cross into "covered"). These are the essential
   *   second/third passes — a frame that only delivers these still adds
   *   real information and must not be discarded.
   * - `wouldCover`: subset of `advancesPartial` that would reach the
   *   covered threshold on this observation.
   */
  peek(pose: CameraPose): {
    newlyTouched: number;
    advancesPartial: number;
    wouldCover: number;
    observedKeys: VoxelKey[];
  } {
    const keys = this.sampleFrustumVoxels(pose);
    const seen = new Set<VoxelKey>();
    let newlyTouched = 0;
    let advancesPartial = 0;
    let wouldCover = 0;
    for (const { key } of keys) {
      if (seen.has(key)) continue;
      seen.add(key);
      const entry = this.voxels.get(key);
      if (!entry) {
        newlyTouched += 1;
        continue;
      }
      if (entry.observationCount < this.config.minObservations) {
        advancesPartial += 1;
        if (entry.observationCount + 1 >= this.config.minObservations) {
          wouldCover += 1;
        }
      }
    }
    return { newlyTouched, advancesPartial, wouldCover, observedKeys: Array.from(seen) };
  }

  /** covered / touched, in [0,1]. Returns 0 if nothing observed yet. */
  get coveragePercent(): number {
    if (this.voxels.size === 0) return 0;
    return this.coveredCount / this.voxels.size;
  }

  get touchedVoxelCount(): number {
    return this.voxels.size;
  }

  get coveredVoxelCount(): number {
    return this.coveredCount;
  }

  /** Snapshot of all touched voxels, for AR overlay rendering. */
  getVoxels(): VoxelView[] {
    const out: VoxelView[] = [];
    const s = this.config.voxelSize;
    for (const [key, entry] of this.voxels.entries()) {
      out.push({
        key,
        center: [
          (entry.ix + 0.5) * s,
          (entry.iy + 0.5) * s,
          (entry.iz + 0.5) * s,
        ],
        state: entry.observationCount >= this.config.minObservations ? 'covered' : 'partial',
        observationCount: entry.observationCount,
      });
    }
    return out;
  }

  reset(): void {
    this.voxels.clear();
    this.coveredCount = 0;
  }

  // ─── internals ────────────────────────────────────────────

  /**
   * Sample points inside the camera frustum and snap each to its voxel.
   *
   * Strategy: walk along the camera forward axis in voxel-sized depth
   * steps. At each depth, the visible cross-section is a rectangle whose
   * size grows with tan(halfFov) * depth. Sample that rectangle on a
   * voxel-sized grid in camera-local space, then project to world.
   *
   * This is O(depth/voxelSize * lateralSamples^2) per call — for our
   * defaults (depth 5m, voxel 0.15m, FOV 60°) that's roughly a few
   * thousand samples, dedup'd into ~hundreds of unique voxels. Fast
   * enough to run on every accepted keyframe (not every camera frame).
   */
  private sampleFrustumVoxels(pose: CameraPose): Array<{ key: VoxelKey; ix: number; iy: number; iz: number }> {
    const { voxelSize, fovDeg, frustumDepth, aspect, nearPlane } = this.config;
    const halfFovH = (fovDeg * Math.PI) / 180 / 2;
    const halfFovV = Math.atan(Math.tan(halfFovH) / aspect);

    const forward = normalize(pose.forward);
    // Build orthonormal basis. `up` from pose may not be perpendicular
    // to forward after IMU noise — re-orthogonalise.
    const upHint = normalize(pose.up);
    const right = normalize(cross(forward, upHint));
    const up = normalize(cross(right, forward));

    const results: Array<{ key: VoxelKey; ix: number; iy: number; iz: number }> = [];
    const seen = new Set<VoxelKey>();

    const depthStep = voxelSize;
    const start = Math.max(nearPlane, depthStep);
    for (let z = start; z <= frustumDepth; z += depthStep) {
      const halfW = Math.tan(halfFovH) * z;
      const halfH = Math.tan(halfFovV) * z;
      // Lateral step = voxelSize so we never skip a voxel-width gap.
      for (let x = -halfW; x <= halfW; x += voxelSize) {
        for (let y = -halfH; y <= halfH; y += voxelSize) {
          // Project local (x, y, z) to world: origin + x*right + y*up + z*forward
          const wx = pose.position[0] + right[0] * x + up[0] * y + forward[0] * z;
          const wy = pose.position[1] + right[1] * x + up[1] * y + forward[1] * z;
          const wz = pose.position[2] + right[2] * x + up[2] * y + forward[2] * z;
          const ix = Math.floor(wx / voxelSize);
          const iy = Math.floor(wy / voxelSize);
          const iz = Math.floor(wz / voxelSize);
          const key = `${ix}_${iy}_${iz}`;
          if (seen.has(key)) continue;
          seen.add(key);
          results.push({ key, ix, iy, iz });
        }
      }
    }
    return results;
  }
}

// ─── helpers exported for tests / overlay ─────────────────────

/** Decode a VoxelKey like "1_-2_3" into its integer coordinates. */
export function parseVoxelKey(key: VoxelKey): [number, number, number] {
  const parts = key.split('_');
  return [Number(parts[0]), Number(parts[1]), Number(parts[2])];
}

/** True if a is approximately on the same side of b as `forward`. */
export function isInFront(a: Vec3, b: Vec3, forward: Vec3): boolean {
  const delta = subtract(a, b);
  if (vecLength(delta) === 0) return false;
  return dot(normalize(delta), normalize(forward)) > 0;
}
