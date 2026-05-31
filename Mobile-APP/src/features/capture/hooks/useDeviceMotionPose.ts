// ────────────────────────────────────────────────────────────
// useDeviceMotionPose — feed phone orientation into SessionProvider
// ────────────────────────────────────────────────────────────
//
// CaptureScreen uses VisionCamera which has no AR tracking, so the
// AngleGate has nothing to gate on. This hook uses the phone's IMU
// (gyroscope + accelerometer fused by Expo) to produce a rotation-only
// CameraPose at ~10 Hz. Position stays [0,0,0] — AngleGate's distance
// term is unused, only the forward-vector cosine similarity matters.
//
// When the user pivots away from a previously-saved direction, AngleGate
// starts allowing captures again. When they hold steady on the same
// direction, captures are skipped. That's the whole win.
// ────────────────────────────────────────────────────────────

import { useEffect } from 'react';
import { DeviceMotion } from 'expo-sensors';
import { CameraPose } from '../../../shared/core/types';

interface UseDeviceMotionPoseOptions {
  /** Called with each new pose. Throttled by `intervalMs`. */
  onPose: (pose: CameraPose) => void;
  /** Whether the subscription is active. */
  enabled: boolean;
  /** Sensor update interval in ms (default 100 = 10 Hz). */
  intervalMs?: number;
}

/**
 * Build a rotation matrix from intrinsic Z-X-Y Euler angles (alpha, beta, gamma)
 * as supplied by expo-sensors DeviceMotion, then extract the world-space
 * "forward" and "up" unit vectors for the back-facing camera.
 *
 * Device frame: +X right, +Y up, -Z out the back camera.
 * Back camera "forward" in device frame = (0, 0, -1).
 */
function rotationToForwardUp(
  alpha: number,
  beta: number,
  gamma: number,
): { forward: [number, number, number]; up: [number, number, number] } {
  const ca = Math.cos(alpha), sa = Math.sin(alpha);
  const cb = Math.cos(beta), sb = Math.sin(beta);
  const cg = Math.cos(gamma), sg = Math.sin(gamma);

  // R = Rz(alpha) * Rx(beta) * Ry(gamma) — Z-X-Y intrinsic.
  // Standard composition; columns are the rotated basis vectors in world space.
  const m00 = ca * cg - sa * sb * sg;
  const m01 = -sa * cb;
  const m02 = ca * sg + sa * sb * cg;
  const m10 = sa * cg + ca * sb * sg;
  const m11 = ca * cb;
  const m12 = sa * sg - ca * sb * cg;
  const m20 = -cb * sg;
  const m21 = sb;
  const m22 = cb * cg;

  // forward_device = (0,0,-1) → world = -3rd column of R
  const forward: [number, number, number] = [-m02, -m12, -m22];
  // up_device = (0,1,0) → world = 2nd column of R
  const up: [number, number, number] = [m01, m11, m21];
  return { forward, up };
}

export function useDeviceMotionPose({
  onPose,
  enabled,
  intervalMs = 100,
}: UseDeviceMotionPoseOptions): void {
  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let subscription: ReturnType<typeof DeviceMotion.addListener> | null = null;

    (async () => {
      const available = await DeviceMotion.isAvailableAsync();
      if (!available || cancelled) return;

      DeviceMotion.setUpdateInterval(intervalMs);
      subscription = DeviceMotion.addListener((data) => {
        const r = data.rotation;
        if (!r) return;
        const { forward, up } = rotationToForwardUp(r.alpha, r.beta, r.gamma);
        onPose({
          position: [0, 0, 0], // unknown — sensor fusion can't recover position
          rotation: [r.beta, r.gamma, r.alpha],
          forward,
          up,
          timestamp: Date.now(),
        });
      });
    })();

    return () => {
      cancelled = true;
      subscription?.remove();
    };
  }, [enabled, intervalMs, onPose]);
}
