// ────────────────────────────────────────────────────────────
// CoverageDemoScene — ViroARScene that drives the CoverageTracker
// ────────────────────────────────────────────────────────────
//
// Runs an AR hit-test along the camera's centre ray each frame and
// forwards the best surface hit point (throttled) to the screen, which
// marks the voxel containing it. This coats real geometry instead of
// filling the camera frustum with floating cubes.
//
// Hit-point flow:
//   ViroARScene.onCameraARHitTest
//     → pick best hit (plane > feature point) along the centre ray
//     → onHitPoint(worldPoint)   [throttled to ~5 Hz]
//     → CoverageDemoScreen.handleHitPoint: tracker.observePoint(point)
//     → new `voxels` flow back in via viroAppProps so the overlay re-renders
// ────────────────────────────────────────────────────────────

import React, { useRef, useCallback } from 'react';
import { ViroARScene } from '@reactvision/react-viro';
import VoxelOverlay from './VoxelOverlay';
import { VoxelView } from './CoverageTracker';
import { Vec3 } from '../../shared/core/types';

interface SceneProps {
  sceneNavigator: {
    viroAppProps: {
      voxels: VoxelView[];
      voxelSize: number;
      onHitPoint: (point: Vec3) => void;
      onTracking?: (state: 'unavailable' | 'limited' | 'normal') => void;
      /** Minimum ms between forwarded hit points (default ~200 = 5 Hz). */
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

export default function CoverageDemoScene(props: SceneProps) {
  const { voxels, voxelSize, onHitPoint, onTracking, hitIntervalMs = 200 } =
    props.sceneNavigator.viroAppProps;

  const lastEmitRef = useRef(0);
  // Only mark coverage when tracking is fully initialised.
  const trackingOkRef = useRef(false);

  const handleCameraARHitTest = useCallback(
    (results: any) => {
      if (!trackingOkRef.current) return; // ignore hits during limited/unavailable
      const now = Date.now();
      if (now - lastEmitRef.current < hitIntervalMs) return;

      const hits: any[] = results?.hitTestResults ?? [];
      if (hits.length === 0) return;

      // Pick the highest-confidence hit along the camera's centre ray.
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

      lastEmitRef.current = now;
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
      onCameraARHitTest={handleCameraARHitTest}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
    >
      <VoxelOverlay voxels={voxels} voxelSize={voxelSize} />
    </ViroARScene>
  );
}
