import React, { useRef, useCallback, useState, useEffect } from 'react';
import {
  ViroARScene,
  Viro3DObject,
  ViroNode,
  ViroAmbientLight,
  ViroDirectionalLight,
  ViroMaterials,
} from '@reactvision/react-viro';
import ARReticle from './ARReticle';

const TRACKING_NORMAL = 3;
const TRACKING_LIMITED = 2;

const HIT_PRIORITY: Record<string, number> = {
  ExistingPlaneUsingExtent: 4,
  ExistingPlane: 3,
  EstimatedHorizontalPlane: 2,
  FeaturePoint: 1,
  DepthPoint: 0,
};

ViroMaterials.createMaterials({
  reticleMaterial: {
    diffuseColor: '#ffffff',
    diffuseIntensity: 0.8,
  },
});

export default function ARMeshScene(arSceneProps: any) {
  const props = arSceneProps.sceneNavigator?.viroAppProps ?? arSceneProps;

  const meshSource: any = props.meshSource;
  const meshType: 'GLB' | 'OBJ' = props.meshType ?? 'GLB';
  const interactionMode: string = props.interactionMode ?? 'place';
  const onTrackingChanged: ((state: string) => void) | undefined = props.onTrackingChanged;
  const onMeshPlaced: ((placed: boolean) => void) | undefined = props.onMeshPlaced;
  const onMeshLoading: ((loading: boolean) => void) | undefined = props.onMeshLoading;
  const onReticleVisible: ((visible: boolean) => void) | undefined = props.onReticleVisible;

  const [isMeshPlaced, setIsMeshPlaced] = useState(false);
  const [meshPosition, setMeshPosition] = useState<[number, number, number]>([0, 0, -1]);
  const [meshRotation, setMeshRotation] = useState<[number, number, number]>([0, 0, 0]);
  const [meshScale, setMeshScale] = useState<[number, number, number]>([0.2, 0.2, 0.2]);
  const [planeAnchorPos, setPlaneAnchorPos] = useState<[number, number, number]>([0, 0, -1]);

  const [reticlePosition, setReticlePosition] = useState<[number, number, number]>([0, 0, -1]);
  const [reticleVisible, setReticleVisible] = useState(false);

  const currentRotationY = useRef(0);
  const latestHitPosition = useRef<[number, number, number] | null>(null);

  const handleTrackingUpdated = useCallback(
    (state: number) => {
      if (onTrackingChanged) {
        if (state === TRACKING_NORMAL) {
          onTrackingChanged('normal');
        } else if (state === TRACKING_LIMITED) {
          onTrackingChanged('limited');
        } else {
          onTrackingChanged('unavailable');
        }
      }
    },
    [onTrackingChanged],
  );

  const handleCameraARHitTest = useCallback(
    (results: any) => {
      if (isMeshPlaced) return;

      const hitTestResults: any[] = results?.hitTestResults ?? [];

      if (hitTestResults.length === 0) {
        setReticleVisible(false);
        latestHitPosition.current = null;
        if (onReticleVisible) onReticleVisible(false);
        return;
      }

      let bestHit: any = null;
      let bestPriority = -1;

      for (const hit of hitTestResults) {
        const priority = HIT_PRIORITY[hit.type] ?? -1;
        if (priority > bestPriority) {
          bestPriority = priority;
          bestHit = hit;
        }
      }

      if (bestHit && bestPriority >= 2) {
        const pos: [number, number, number] = [
          bestHit.transform?.position?.[0] ?? 0,
          bestHit.transform?.position?.[1] ?? 0,
          bestHit.transform?.position?.[2] ?? -1,
        ];
        setReticlePosition(pos);
        setReticleVisible(true);
        latestHitPosition.current = pos;
        if (onReticleVisible) onReticleVisible(true);
      } else {
        setReticleVisible(false);
        latestHitPosition.current = null;
        if (onReticleVisible) onReticleVisible(false);
      }
    },
    [isMeshPlaced, onReticleVisible],
  );

  const handleClick = useCallback(
    (_position: any) => {
      if (isMeshPlaced || !latestHitPosition.current) return;

      const pos = latestHitPosition.current;
      setMeshPosition(pos);
      setPlaneAnchorPos(pos);
      setIsMeshPlaced(true);
      setReticleVisible(false);
      if (onMeshPlaced) onMeshPlaced(true);
    },
    [isMeshPlaced, onMeshPlaced],
  );

  const handleDrag = useCallback((dragToPos: number[]) => {
    setMeshPosition([dragToPos[0] ?? 0, dragToPos[1] ?? 0, dragToPos[2] ?? 0]);
  }, []);

  const handleRotate = useCallback((rotateState: number, rotationFactor: number, _source: any) => {
    if (rotateState === 2) {
      currentRotationY.current += rotationFactor * 0.5;
      setMeshRotation([0, currentRotationY.current, 0]);
    }
  }, []);

  const handlePinch = useCallback((pinchState: number, scaleFactor: number, _source: any) => {
    if (pinchState === 2) {
      setMeshScale((prev) => {
        const newScale = Math.max(0.01, Math.min(5, prev[0] * scaleFactor));
        return [newScale, newScale, newScale];
      });
    }
  }, []);

  const enableDrag = interactionMode === 'move' && isMeshPlaced;
  const enableRotate = interactionMode === 'rotate' && isMeshPlaced;
  const enablePinch = interactionMode === 'scale' && isMeshPlaced;

  useEffect(() => {
    if (props.resetRequested) {
      setIsMeshPlaced(false);
      setMeshPosition([0, 0, -1]);
      setMeshRotation([0, 0, 0]);
      setMeshScale([0.2, 0.2, 0.2]);
      currentRotationY.current = 0;
      setReticleVisible(false);
      latestHitPosition.current = null;
    }
  }, [props.resetRequested]);

  return (
    <ViroARScene
      onTrackingUpdated={handleTrackingUpdated}
      onCameraARHitTest={handleCameraARHitTest}
      onClick={handleClick}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
      displayPointCloud={{
        imageSource: require('../../../assets/meshes/point_cloud_point.png'),
        imageScale: [0.02, 0.02, 0.02],
        maxPoints: 200,
      }}
    >
      <ViroAmbientLight color="#FFFFFF" intensity={500} />
      <ViroDirectionalLight
        direction={[0, -1, -0.5]}
        castsShadow={true}
        shadowOrthographicPosition={[0, 3, -2]}
        shadowOrthographicSize={5}
        shadowBias={0.003}
        color="#FFFFFF"
        intensity={800}
      />

      {!isMeshPlaced && <ARReticle position={reticlePosition} visible={reticleVisible} />}

      {isMeshPlaced && meshSource && (
        <ViroNode
          position={meshPosition}
          rotation={meshRotation}
          scale={meshScale}
          dragType={enableDrag ? 'FixedToPlane' : 'FixedDistance'}
          dragPlane={
            enableDrag
              ? {
                  planePoint: planeAnchorPos,
                  planeNormal: [0, 1, 0],
                  maxDistance: 5,
                }
              : undefined
          }
          onDrag={enableDrag ? handleDrag : undefined}
          onRotate={enableRotate ? handleRotate : undefined}
          onPinch={enablePinch ? handlePinch : undefined}
        >
          <Viro3DObject
            source={meshSource}
            type={meshType}
            position={[0, 0, 0]}
            scale={[1, 1, 1]}
            onLoadStart={() => {
              if (onMeshLoading) onMeshLoading(true);
            }}
            onLoadEnd={() => {
              if (onMeshLoading) onMeshLoading(false);
            }}
            onError={(e: any) => console.warn('Mesh load error:', e.nativeEvent?.error)}
          />
        </ViroNode>
      )}
    </ViroARScene>
  );
}
