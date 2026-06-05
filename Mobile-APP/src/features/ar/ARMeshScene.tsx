import React, { useRef, useCallback, useState, useEffect } from 'react';
import {
  ViroARScene,
  Viro3DObject,
  ViroNode,
  ViroAmbientLight,
  ViroMaterials,
} from '@reactvision/react-viro';
import ARReticle from './ARReticle';
import type { PlacedMesh } from './arTypes';

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
  selectionRing: {
    diffuseColor: 'rgba(80, 210, 255, 0.45)',
    lightingModel: 'Constant',
    blendMode: 'Alpha',
  },
});

export default function ARMeshScene(arSceneProps: any) {
  const props = arSceneProps.sceneNavigator?.viroAppProps ?? arSceneProps;

  const meshes: PlacedMesh[] = props.meshes ?? [];
  const activeMeshId: string | null = props.activeMeshId ?? null;
  const interactionMode: string = props.interactionMode ?? 'select';
  const onTrackingChanged: ((state: string) => void) | undefined = props.onTrackingChanged;
  const onMeshPlaced: ((id: string) => void) | undefined = props.onMeshPlaced;
  const onMeshSelected: ((id: string) => void) | undefined = props.onMeshSelected;
  const onMeshLoading: ((loading: boolean) => void) | undefined = props.onMeshLoading;
  const onReticleVisible: ((visible: boolean) => void) | undefined = props.onReticleVisible;
  const onMeshPlacedExt: ((id: string, position: [number, number, number]) => void) | undefined = props.onMeshPlacedExt;
  const onMeshMoved: ((id: string, position: [number, number, number]) => void) | undefined = props.onMeshMoved;
  const onCameraPoseUpdate: ((pose: any) => void) | undefined = props.onCameraPoseUpdate;

  // Scene-local positions (keyed by mesh id) — avoids parent re-renders on every drag
  const [positions, setPositions] = useState<Record<string, [number, number, number]>>(props.initialPositions ?? {});
  const [planeAnchors, setPlaneAnchors] = useState<Record<string, [number, number, number]>>(props.initialPositions ?? {});

  const [reticlePosition, setReticlePosition] = useState<[number, number, number]>([0, 0, -1]);
  const [reticleVisible, setReticleVisible] = useState(false);
  const latestHitPosition = useRef<[number, number, number] | null>(null);

  const hasUnplacedMesh = meshes.some((m) => !m.isPlaced);
  const unplacedMesh = meshes.find((m) => !m.isPlaced) ?? null;

  // Reset when parent requests it
  useEffect(() => {
    if (props.resetRequested) {
      setPositions(props.initialPositions ?? {});
      setPlaneAnchors(props.initialPositions ?? {});
      setReticlePosition([0, 0, -1]);
      setReticleVisible(false);
      latestHitPosition.current = null;
    }
  }, [props.resetRequested, props.initialPositions]);

  // ─── Viro callbacks ────────────────────────────────────────────────────────

  const handleTrackingUpdated = useCallback(
    (state: number) => {
      if (!onTrackingChanged) return;
      if (state === TRACKING_NORMAL) onTrackingChanged('normal');
      else if (state === TRACKING_LIMITED) onTrackingChanged('limited');
      else onTrackingChanged('unavailable');
    },
    [onTrackingChanged],
  );

  const handleCameraARHitTest = useCallback(
    (results: any) => {
      if (!hasUnplacedMesh) return;

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
    [hasUnplacedMesh, onReticleVisible],
  );

  /** Tap on the AR scene background — places the pending unplaced mesh */
  const handleSceneClick = useCallback(() => {
    if (!unplacedMesh || !latestHitPosition.current) return;
    const pos = latestHitPosition.current;
    setPositions((prev) => ({ ...prev, [unplacedMesh.id]: pos }));
    setPlaneAnchors((prev) => ({ ...prev, [unplacedMesh.id]: pos }));
    if (onMeshPlaced) onMeshPlaced(unplacedMesh.id);
    if (onMeshPlacedExt) onMeshPlacedExt(unplacedMesh.id, pos);
    setReticleVisible(false);
    latestHitPosition.current = null;
  }, [unplacedMesh, onMeshPlaced, onMeshPlacedExt]);

  return (
    <ViroARScene
      onTrackingUpdated={handleTrackingUpdated}
      onCameraARHitTest={handleCameraARHitTest}
      onClick={handleSceneClick}
      onCameraTransformUpdate={(e: any) => {
        if (onCameraPoseUpdate) onCameraPoseUpdate(e.cameraTransform);
      }}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
    >
      <ViroAmbientLight color="#FFFFFF" intensity={1000} />

      {hasUnplacedMesh && <ARReticle position={reticlePosition} visible={reticleVisible} />}

      {meshes.map((mesh) => {
        if (!mesh.isPlaced || !mesh.meshSource) return null;

        const isActive = mesh.id === activeMeshId;
        const meshPos = positions[mesh.id] ?? props.initialPositions?.[mesh.id] ?? [0, 0, -1];
        const planeAnchor = planeAnchors[mesh.id] ?? props.initialPositions?.[mesh.id] ?? meshPos;
        const enableDrag =
          isActive && (interactionMode === 'move-floor' || interactionMode === 'move-lift');

        return (
          <ViroNode
            key={mesh.id}
            position={meshPos}
            rotation={mesh.rotation}
            scale={mesh.scale}
            dragType={
              enableDrag
                ? interactionMode === 'move-lift'
                  ? 'FixedDistance'
                  : 'FixedToPlane'
                : undefined
            }
            dragPlane={
              enableDrag && interactionMode === 'move-floor'
                ? {
                    planePoint: planeAnchor,
                    planeNormal: [0, 1, 0],
                    maxDistance: 5,
                  }
                : undefined
            }
            onDrag={
              enableDrag
                ? (dragToPos: number[]) => {
                    const newPos: [number, number, number] = [dragToPos[0] ?? 0, dragToPos[1] ?? 0, dragToPos[2] ?? 0];
                    setPositions((prev) => ({
                      ...prev,
                      [mesh.id]: newPos,
                    }));
                    if (onMeshMoved) onMeshMoved(mesh.id, newPos);
                  }
                : undefined
            }
            onClick={
              interactionMode === 'select'
                ? () => {
                    if (onMeshSelected) onMeshSelected(mesh.id);
                  }
                : undefined
            }
          >
            <Viro3DObject
              source={mesh.meshSource}
              type={mesh.meshType}
              position={[0, 0, 0]}
              scale={[1, 1, 1]}
              onLoadStart={() => {
                console.log('Viro3DObject onLoadStart:', mesh.meshName);
                if (onMeshLoading) onMeshLoading(true);
              }}
              onLoadEnd={() => {
                if (onMeshLoading) onMeshLoading(false);
              }}
              onError={(e: any) =>
                console.warn('Mesh load error:', e.nativeEvent?.error, mesh.meshName)
              }
            />
          </ViroNode>
        );
      })}
    </ViroARScene>
  );
}
