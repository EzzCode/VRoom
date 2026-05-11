// ────────────────────────────────────────────────────────────
// Strategy Pattern: ICaptureGate
// ────────────────────────────────────────────────────────────
import { CameraPose, GateResult } from '../../../shared/core/types';

/**
 * A capture gate decides whether a candidate frame should be saved as a keyframe.
 *
 * The KeyframeExtractor iterates through all registered gates — a frame is
 * saved only if **every** gate passes. This follows the Strategy Pattern and
 * is open for extension (add an ExposureGate, MotionBlurGate, etc.) without
 * modifying existing code.
 */
export interface ICaptureGate {
  /** Human-readable gate name (used in HUD prompts) */
  readonly name: string;

  /**
   * Evaluate whether the current frame should pass this gate.
   *
   * @param pose  - The current camera pose from ARKit/ARCore (null in Build 1 when AR isn't active yet)
   * @returns       GateResult indicating whether the gate passed + optional guidance text
   */
  evaluate(pose: CameraPose | null): GateResult;

  /** Reset any internal state (called on session start) */
  reset(): void;
}
