import React, { useCallback, useRef } from 'react';
import { ViroARScene } from '@reactvision/react-viro';
import { invertRigidTransform, makeCameraToWorld, rebaseToSessionRoot } from '../../shared/core/arPose';
import { Matrix4, TrackingState, Vec3 } from '../../shared/core/types';

type CameraTransform = {
  position: Vec3;
  rotation: Vec3;
  forward: Vec3;
  up: Vec3;
};

type Props = {
  onTrackingChanged?: (state: TrackingState) => void;
  onCameraPose?: (pose: {
    position: Vec3;
    rotation: Vec3;
    forward: Vec3;
    up: Vec3;
    cameraToWorld: Matrix4;
  }) => void;
};

const TRACKING_NORMAL = 3;
const TRACKING_LIMITED = 2;

export default function ARCaptureScene(arSceneProps: any) {
  const props: Props = arSceneProps.sceneNavigator?.viroAppProps ?? arSceneProps;
  const rootInverseRef = useRef<Matrix4 | null>(null);

  const handleTrackingUpdated = useCallback(
    (state: number) => {
      if (state === TRACKING_NORMAL) {
        props.onTrackingChanged?.('normal');
      } else if (state === TRACKING_LIMITED) {
        props.onTrackingChanged?.('limited');
      } else {
        props.onTrackingChanged?.('unavailable');
      }
    },
    [props],
  );

  const handleCameraTransformUpdate = useCallback(
    (transform: CameraTransform) => {
      const absoluteCameraToWorld = makeCameraToWorld(
        transform.position,
        transform.forward,
        transform.up,
      );

      if (rootInverseRef.current == null) {
        rootInverseRef.current = invertRigidTransform(absoluteCameraToWorld);
      }

      props.onCameraPose?.({
        position: transform.position,
        rotation: transform.rotation,
        forward: transform.forward,
        up: transform.up,
        cameraToWorld: rebaseToSessionRoot(rootInverseRef.current, absoluteCameraToWorld),
      });
    },
    [props],
  );

  return (
    <ViroARScene
      onTrackingUpdated={handleTrackingUpdated}
      onCameraTransformUpdate={handleCameraTransformUpdate}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
    />
  );
}
