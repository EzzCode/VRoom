// ────────────────────────────────────────────────────────────
// KeyframeExtractor — Orchestrates all capture gates
// ────────────────────────────────────────────────────────────
import { ICaptureGate } from './gates/ICaptureGate';
import { CameraPose, GateResult } from '../../shared/core/types';

/** A single gate's evaluation entry returned by KeyframeExtractor.evaluate() */
export interface GateEvalEntry {
  gate: string;
  result: GateResult;
}

/**
 * The KeyframeExtractor runs every registered gate in sequence.
 *
 * A frame becomes a keyframe ONLY if ALL gates pass.
 * If any gate fails, its guidance reason is surfaced to the HUD.
 *
 * This is the Strategy Pattern orchestrator — gates can be added or
 * removed at runtime without touching this class.
 */
export class KeyframeExtractor {
  private gates: ICaptureGate[] = [];

  /** Register a new gate into the pipeline */
  addGate(gate: ICaptureGate): void {
    this.gates.push(gate);
  }

  /** Remove a gate by name */
  removeGate(name: string): void {
    this.gates = this.gates.filter((g) => g.name !== name);
  }

  /**
   * Evaluate all gates against the current frame context.
   *
   * @param pose - Camera pose from AR (null if AR isn't active)
   * @returns Object with overall pass/fail and array of individual results
   */
  evaluate(pose: CameraPose | null): {
    shouldCapture: boolean;
    results: GateEvalEntry[];
  } {
    const results: GateEvalEntry[] = [];
    let shouldCapture = true;

    for (const gate of this.gates) {
      const result = gate.evaluate(pose);
      results.push({ gate: gate.name, result });
      if (!result.passed) {
        shouldCapture = false;
        // Don't break — evaluate all gates so we can show all guidance
      }
    }

    return { shouldCapture, results };
  }

  /** Reset all gates (called at session start) */
  resetAll(): void {
    for (const gate of this.gates) {
      gate.reset();
    }
  }

  /** List registered gate names */
  get gateNames(): string[] {
    return this.gates.map((g) => g.name);
  }
}
