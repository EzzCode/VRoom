// ────────────────────────────────────────────────────────────
// CoverageDemoScene — ViroARScene that drives the CoverageTracker
// ────────────────────────────────────────────────────────────
//
// Subscribes to ARKit/ARCore camera transform updates, throttles them,
// and forwards them to the SessionProvider via the `viroAppProps` callbacks.
// Renders the live VoxelOverlay on top of the AR camera feed.
//
// Pose flow:
//   ViroARScene.onCameraTransformUpdate
//     → onPose(pose)    [throttled to ~5 Hz]
//     → SessionProvider.setCurrentPose + coverageTracker.observe(pose)
//     → forceTick() so the overlay re-renders with new voxels
// ────────────────────────────────────────────────────────────

import React, { useRef, useCallback } from 'react';
import { ViroARScene } from '@reactvision/react-viro';
import VoxelOverlay from './VoxelOverlay';
import { VoxelView } from './CoverageTracker';
import { CameraPose } from '../../shared/core/types';

interface SceneProps {
  sceneNavigator: {
    viroAppProps: {
      voxels: VoxelView[];
      voxelSize: number;
      onPose: (pose: CameraPose) => void;
      onTracking?: (state: 'unavailable' | 'limited' | 'normal') => void;
      /** Minimum ms between forwarded poses (default ~200 = 5 Hz). */
      poseIntervalMs?: number;
    };
  };
}

const TRACKING_NORMAL = 3;
const TRACKING_LIMITED = 2;

export default function CoverageDemoScene(props: SceneProps) {
  const { voxels, voxelSize, onPose, onTracking, poseIntervalMs = 200 } =
    props.sceneNavigator.viroAppProps;

  const lastEmitRef = useRef(0);
  // Only forward poses when tracking is fully initialised.
  const trackingOkRef = useRef(false);

  const handleCameraTransformUpdate = useCallback(
    (transform: any) => {
      if (!trackingOkRef.current) return;          // drop poses during limited/unavailable
      const now = Date.now();
      if (now - lastEmitRef.current < poseIntervalMs) return;
      lastEmitRef.current = now;

      // ViroCameraTransform exposes both legacy (cameraTransform.*) and flat keys.
      const position = transform.position ?? transform.cameraTransform?.position;
      const rotation = transform.rotation ?? transform.cameraTransform?.rotation;
      const forward = transform.forward ?? transform.cameraTransform?.forward;
      const up = transform.up ?? transform.cameraTransform?.up;
      if (!position || !forward || !up) return;

      onPose({
        position: [position[0], position[1], position[2]],
        rotation: rotation
          ? [rotation[0], rotation[1], rotation[2]]
          : [0, 0, 0],
        forward: [forward[0], forward[1], forward[2]],
        up: [up[0], up[1], up[2]],
        timestamp: now,
      });
    },
    [onPose, poseIntervalMs],
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
      onCameraTransformUpdate={handleCameraTransformUpdate}
      onTrackingUpdated={handleTrackingUpdated}
    >
      <VoxelOverlay voxels={voxels} voxelSize={voxelSize} />
    </ViroARScene>
  );
}
