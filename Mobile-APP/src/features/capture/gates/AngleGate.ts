// ────────────────────────────────────────────────────────────
// Strategy: Angle Diversity Gate
// ────────────────────────────────────────────────────────────
import { ICaptureGate } from './ICaptureGate';
import { CameraPose, GateResult } from '../../../shared/core/types';
import { distance, cosineSimilarity } from '../../../shared/core/math';
import { CAPTURE_CONFIG } from '../config/captureConfig';

/**
 * Prevents redundant captures from nearly identical viewpoints.
 *
 * For each new candidate:
 *   1. Compute Euclidean distance to every saved pose
 *   2. Compute cosine similarity of forward-direction vectors
 *   3. Block capture if BOTH distance < minDistance AND similarity > maxSimilarity
 *
 * This forces the user to either move physically or rotate substantially.
 */
export class AngleGate implements ICaptureGate {
  readonly name = 'AngleGate';

  private savedPoses: CameraPose[] = [];
  private readonly minDistance: number;
  private readonly maxSimilarity: number;

  constructor(
    minDistance = CAPTURE_CONFIG.angleDiversity.minDistance,
    maxSimilarity = CAPTURE_CONFIG.angleDiversity.maxSimilarity,
  ) {
    this.minDistance = minDistance;
    this.maxSimilarity = maxSimilarity;
  }

  evaluate(pose: CameraPose | null): GateResult {
    // If AR pose isn't available yet, pass by default
    if (!pose) {
      return { passed: true };
    }

    // Check against all saved poses
    for (const saved of this.savedPoses) {
      const dist = distance(pose.position, saved.position);
      const sim = cosineSimilarity(pose.forward, saved.forward);

      if (dist < this.minDistance && sim > this.maxSimilarity) {
        return {
          passed: false,
          reason: 'Move sideways or rotate to capture a new angle.',
        };
      }
    }

    // Passed — record this pose
    this.savedPoses.push({ ...pose });
    return { passed: true };
  }

  reset(): void {
    this.savedPoses = [];
  }

  /** Returns the number of unique angles saved so far */
  get poseCount(): number {
    return this.savedPoses.length;
  }

  /** Returns a read-only copy of all saved poses */
  get poses(): ReadonlyArray<CameraPose> {
    return this.savedPoses;
  }
}
