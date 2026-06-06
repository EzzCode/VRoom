// ────────────────────────────────────────────────────────────
// ARCaptureScene — ViroARScene driving the real capture session
// ────────────────────────────────────────────────────────────
//
// Unlike the VisionCamera capture path (rotation-only IMU pose +
// frustum-volume coverage), this scene runs on the AR camera and provides:
//   - real 6DoF pose via onCameraTransformUpdate  → feeds AngleGate + keyframes
//   - surface hit points via onCameraARHitTest     → feeds coverage (observePoint)
// It also renders the live VoxelOverlay so the user sees coverage build up.
//
// Both callbacks are throttled and gated on tracking == normal so we don't
// pollute pose/coverage during ARKit/ARCore initialisation.
// ────────────────────────────────────────────────────────────

import React, { useRef, useCallback } from 'react';
import { ViroARScene } from '@reactvision/react-viro';
import VoxelOverlay from '../../coverage/VoxelOverlay';
import { VoxelView } from '../../coverage/CoverageTracker';
import { CameraPose, Vec3 } from '../../../shared/core/types';

interface SceneProps {
  sceneNavigator: {
    viroAppProps: {
      voxels: VoxelView[];
      voxelSize: number;
      /** Real 6DoF camera pose (throttled). */
      onPose: (pose: CameraPose) => void;
      /** Best surface hit point along the camera centre ray (throttled). */
      onHitPoint: (point: Vec3) => void;
      onTracking?: (state: 'unavailable' | 'limited' | 'normal') => void;
      /** Min ms between forwarded poses (default ~150 = ~6-7 Hz). */
      poseIntervalMs?: number;
      /** Min ms between forwarded hit points (default ~150). */
      hitIntervalMs?: number;
    };
  };
}

const TRACKING_NORMAL = 3;
const TRACKING_LIMITED = 2;

// Prefer real planes over estimated planes over raw feature points.
const HIT_PRIORITY: Record<string, number> = {
  ExistingPlaneUsingExtent: 4,
  ExistingPlane: 3,
  EstimatedHorizontalPlane: 2,
  FeaturePoint: 1,
  DepthPoint: 0,
};

export default function ARCaptureScene(props: SceneProps) {
  const {
    voxels,
    voxelSize,
    onPose,
    onHitPoint,
    onTracking,
    poseIntervalMs = 150,
    hitIntervalMs = 150,
  } = props.sceneNavigator.viroAppProps;

  const trackingOkRef = useRef(false);
  const lastPoseRef = useRef(0);
  const lastHitRef = useRef(0);

  const handleTransform = useCallback(
    (event: any) => {
      if (!trackingOkRef.current) return;
      const now = Date.now();
      if (now - lastPoseRef.current < poseIntervalMs) return;

      // Viro delivers the pose as event.cameraTransform; fall back to flat keys.
      const t = event?.cameraTransform ?? event;
      const position = t?.position;
      const forward = t?.forward;
      const up = t?.up;
      const rotation = t?.rotation;
      if (!position || !forward || !up) return;

      lastPoseRef.current = now;
      onPose({
        position: [position[0], position[1], position[2]],
        rotation: rotation ? [rotation[0], rotation[1], rotation[2]] : [0, 0, 0],
        forward: [forward[0], forward[1], forward[2]],
        up: [up[0], up[1], up[2]],
        timestamp: now,
      });
    },
    [onPose, poseIntervalMs],
  );

  const handleHitTest = useCallback(
    (results: any) => {
      if (!trackingOkRef.current) return;
      const now = Date.now();
      if (now - lastHitRef.current < hitIntervalMs) return;

      const hits: any[] = results?.hitTestResults ?? [];
      if (hits.length === 0) return;

      let best: any = null;
      let bestPriority = -1;
      for (const hit of hits) {
        const priority = HIT_PRIORITY[hit.type] ?? -1;
        if (priority > bestPriority) {
          bestPriority = priority;
          best = hit;
        }
      }
      const position = best?.transform?.position;
      if (!position) return;

      lastHitRef.current = now;
      onHitPoint([position[0], position[1], position[2]]);
    },
    [onHitPoint, hitIntervalMs],
  );

  const handleTrackingUpdated = useCallback(
    (state: number) => {
      trackingOkRef.current = state === TRACKING_NORMAL;
      if (!onTracking) return;
      if (state === TRACKING_NORMAL) onTracking('normal');
      else if (state === TRACKING_LIMITED) onTracking('limited');
      else onTracking('unavailable');
    },
    [onTracking],
  );

  return (
    <ViroARScene
      onTrackingUpdated={handleTrackingUpdated}
      onCameraTransformUpdate={handleTransform}
      onCameraARHitTest={handleHitTest}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
    >
      <VoxelOverlay voxels={voxels} voxelSize={voxelSize} />
    </ViroARScene>
  );
}
