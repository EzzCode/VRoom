import { ICaptureGate } from './ICaptureGate';
import { CameraPose, GateResult } from '../../../shared/core/types';

/**
 * A gate that allows capturing frames at a fixed time interval.
 */
export class TimeGate implements ICaptureGate {
  readonly name = 'TimeGate';
  private lastCaptureTime: number = 0;
  private readonly intervalMs: number;

  constructor(intervalMs: number = 1000) {
    this.intervalMs = intervalMs;
  }

  evaluate(pose: CameraPose | null): GateResult {
    const now = Date.now();
    if (now - this.lastCaptureTime >= this.intervalMs) {
      this.lastCaptureTime = now;
      return { passed: true };
    }
    return {
      passed: false,
      reason: 'Capturing continuously (1 fps)',
    };
  }

  reset(): void {
    this.lastCaptureTime = 0;
  }
}
