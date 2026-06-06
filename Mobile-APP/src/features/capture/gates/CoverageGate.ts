// ────────────────────────────────────────────────────────────
// Strategy: Coverage Gate
// ────────────────────────────────────────────────────────────
//
// Rejects a candidate frame if the camera frustum at the current pose
// does not add any new voxel observations beyond what we have already.
// This complements AngleGate: AngleGate stops captures from a similar
// pose, CoverageGate stops captures that don't add information (e.g.
// the user pointed at a wall already fully mapped from another angle).
//
// The gate is a thin wrapper around the shared CoverageTracker held by
// the SessionProvider — both the gate and the HUD overlay observe the
// same tracker instance.
// ────────────────────────────────────────────────────────────

import { ICaptureGate } from './ICaptureGate';
import { CameraPose, GateResult } from '../../../shared/core/types';
import { CoverageTracker } from '../../coverage/CoverageTracker';

export class CoverageGate implements ICaptureGate {
  readonly name = 'CoverageGate';

  constructor(
    private readonly tracker: CoverageTracker,
    /** A frame must add at least this many newly touched voxels to pass */
    private readonly minNewVoxels: number = 1,
  ) {}

  evaluate(pose: CameraPose | null): GateResult {
    // No AR pose yet (e.g. tracking not initialised) — let the frame through.
    // BlurGate + AngleGate still apply.
    if (!pose) {
      return { passed: true };
    }

    const { newlyTouched, advancesPartial } = this.tracker.peek(pose);

    // Pass when the frame either reaches new area (newlyTouched) or delivers
    // another observation to a not-yet-covered voxel (advancesPartial). Only
    // reject frames whose entire frustum is already fully covered — those add
    // nothing. (advancesPartial already includes voxels that would cross into
    // "covered" on this pass, so a separate wouldCover check is unnecessary.)
    if (newlyTouched < this.minNewVoxels && advancesPartial === 0) {
      return {
        passed: false,
        reason: 'Aim at an uncovered area to add new coverage.',
      };
    }
    return { passed: true };
  }

  reset(): void {
    this.tracker.reset();
  }
}
