import React, { useRef, useCallback, useState, useEffect } from 'react';
import {
  ViroARScene,
  Viro3DObject,
  ViroNode,
  ViroAmbientLight,
  ViroDirectionalLight,
  ViroQuad,
  ViroMaterials,
  ViroAnimations,
  ViroSpotLight,
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

// Only update the reticle when it moves more than this (metres). Avoids
// re-rendering the whole AR scene on every camera frame (~60 Hz) while a
// mesh is waiting to be placed.
const RETICLE_MOVE_EPSILON = 0.01;

ViroMaterials.createMaterials({
  selectionRing: {
    diffuseColor: 'rgba(80, 210, 255, 0.45)',
    lightingModel: 'Constant',
    blendMode: 'Alpha',
  },
});

ViroAnimations.registerAnimations({
  selectionPulseScaleUp: {
    properties: {
      scaleX: 1.15,
      scaleY: 1.15,
      scaleZ: 1.15,
      opacity: 0.8,
    },
    easing: 'EaseInEaseOut',
    duration: 800,
  },
  selectionPulseScaleDown: {
    properties: {
      scaleX: 0.95,
      scaleY: 0.95,
      scaleZ: 0.95,
      opacity: 0.4,
    },
    easing: 'EaseInEaseOut',
    duration: 800,
  },
  selectionPulseLoop: ['selectionPulseScaleUp', 'selectionPulseScaleDown'] as any,
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
  const onMeshScaled: ((id: string, scale: number) => void) | undefined = props.onMeshScaled;
  const onMeshRotated: ((id: string, rotation: [number, number, number]) => void) | undefined = props.onMeshRotated;
  const onCameraPoseUpdate: ((pose: any) => void) | undefined = props.onCameraPoseUpdate;

  // Scene-local positions (keyed by mesh id) — avoids parent re-renders on every drag
  const [positions, setPositions] = useState<Record<string, [number, number, number]>>(props.initialPositions ?? {});
  const [planeAnchors, setPlaneAnchors] = useState<Record<string, [number, number, number]>>(props.initialPositions ?? {});

  // Ref to always access fresh meshes without causing applyAnchor recreations
  const meshesRef = useRef<PlacedMesh[]>(meshes);
  meshesRef.current = meshes;

  // Base values captured at the start of a pinch/rotate gesture so the
  // multiplicative/additive factors Viro reports are applied from a stable
  // origin instead of compounding every frame.
  const pinchBaseScale = useRef<Record<string, number>>({});
  const rotateBaseYaw = useRef<Record<string, number>>({});

  // ── Real plane-anchor binding ──────────────────────────────────────────────
  // Detected ARKit/ARCore plane anchors, keyed by Viro anchorId. Each placed
  // mesh is bound to the nearest anchor with a fixed offset; when the anchor's
  // transform refines (onAnchorUpdated) we move the mesh by the same offset so
  // it stays locked to the physical surface instead of drifting with the world
  // origin. Layout save/restore still works off world positions (below).
  const arAnchorsRef = useRef<Record<string, { position: [number, number, number] }>>({});
  const meshBindingRef = useRef<
    Record<string, { anchorId: string; offset: [number, number, number] }>
  >({});
  // Mirror of `positions` readable inside anchor callbacks without stale closures.
  const positionsRef = useRef<Record<string, [number, number, number]>>(props.initialPositions ?? {});

  // Calibration offsets to center objects with offsets inside their GLB files
  const [modelOffsets, setModelOffsets] = useState<Record<string, [number, number, number]>>({});
  const [calibratedMeshes, setCalibratedMeshes] = useState<Record<string, boolean>>({});
  const calibrationRefs = useRef<Record<string, any>>({});

  const [reticlePosition, setReticlePosition] = useState<[number, number, number]>([0, 0, -1]);
  const [reticleVisible, setReticleVisible] = useState(false);
  const reticleVisibleRef = useRef(false);
  
  // Keep reticleVisibleRef in sync
  useEffect(() => {
    reticleVisibleRef.current = reticleVisible;
  }, [reticleVisible]);

  const latestHitPosition = useRef<[number, number, number] | null>(null);
  const lastReticleRef = useRef<[number, number, number] | null>(null);

  const hasUnplacedMesh = meshes.some((m) => !m.isPlaced);
  const unplacedMesh = meshes.find((m) => !m.isPlaced) ?? null;

  // Reset when parent requests it
  const prevResetCounterRef = useRef(0);
  useEffect(() => {
    if (props.resetCounter !== undefined && props.resetCounter > prevResetCounterRef.current) {
      prevResetCounterRef.current = props.resetCounter;
      setPositions(props.initialPositions ?? {});
      setPlaneAnchors(props.initialPositions ?? {});
      positionsRef.current = props.initialPositions ?? {};
      meshBindingRef.current = {};
      setModelOffsets({});
      setCalibratedMeshes({});
      calibrationRefs.current = {};
      setReticlePosition([0, 0, -1]);
      setReticleVisible(false);
      latestHitPosition.current = null;
    }
  }, [props.resetCounter, props.initialPositions]);

  // Keep the positions mirror in sync for anchor-update callbacks.
  useEffect(() => {
    positionsRef.current = positions;
  }, [positions]);

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
        latestHitPosition.current = null;
        if (reticleVisibleRef.current) {
          setReticleVisible(false);
          if (onReticleVisible) onReticleVisible(false);
        }
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
        latestHitPosition.current = pos;
        // Only push a state update (re-render) when the reticle actually moved
        // a meaningful amount — otherwise we thrash the scene every frame.
        const prev = lastReticleRef.current;
        const moved =
          !prev ||
          Math.abs(prev[0] - pos[0]) > RETICLE_MOVE_EPSILON ||
          Math.abs(prev[1] - pos[1]) > RETICLE_MOVE_EPSILON ||
          Math.abs(prev[2] - pos[2]) > RETICLE_MOVE_EPSILON;
        if (moved) {
          lastReticleRef.current = pos;
          setReticlePosition(pos);
        }
        if (!reticleVisibleRef.current) {
          setReticleVisible(true);
          if (onReticleVisible) onReticleVisible(true);
        }
      } else {
        latestHitPosition.current = null;
        if (reticleVisibleRef.current) {
          setReticleVisible(false);
          if (onReticleVisible) onReticleVisible(false);
        }
      }
    },
    [hasUnplacedMesh, onReticleVisible],
  );

  // ─── Plane-anchor binding helpers ───────────────────────────────────────────

  /** Nearest detected plane anchor (horizontally) to a world point, or null. */
  const findNearestAnchor = useCallback(
    (worldPos: [number, number, number]): string | null => {
      let bestId: string | null = null;
      let bestDist = Infinity;
      for (const [id, a] of Object.entries(arAnchorsRef.current)) {
        const dx = worldPos[0] - a.position[0];
        const dz = worldPos[2] - a.position[2];
        const dy = Math.abs(worldPos[1] - a.position[1]);
        const horiz = Math.sqrt(dx * dx + dz * dz);
        // Only bind to a plane that is roughly at the object's height and not
        // unreasonably far away.
        if (dy < 0.6 && horiz < 3 && horiz < bestDist) {
          bestDist = horiz;
          bestId = id;
        }
      }
      return bestId;
    },
    [],
  );

  /** Bind a mesh to the nearest plane anchor, storing a fixed offset. */
  const bindMeshToAnchor = useCallback(
    (meshId: string, worldPos: [number, number, number]) => {
      const anchorId = findNearestAnchor(worldPos);
      if (!anchorId) {
        delete meshBindingRef.current[meshId];
        return;
      }
      const a = arAnchorsRef.current[anchorId];
      if (!a) return;
      meshBindingRef.current[meshId] = {
        anchorId,
        offset: [worldPos[0] - a.position[0], worldPos[1] - a.position[1], worldPos[2] - a.position[2]],
      };
    },
    [findNearestAnchor],
  );

  /** Tap on the AR scene background — places the pending unplaced mesh */
  const handleSceneClick = useCallback(() => {
    if (!unplacedMesh || !latestHitPosition.current) return;
    const pos = latestHitPosition.current;
    setPositions((prev) => ({ ...prev, [unplacedMesh.id]: pos }));
    setPlaneAnchors((prev) => ({ ...prev, [unplacedMesh.id]: pos }));
    positionsRef.current[unplacedMesh.id] = pos;
    // Lock the new object to the nearest detected plane anchor so it stays put.
    bindMeshToAnchor(unplacedMesh.id, pos);
    if (onMeshPlaced) onMeshPlaced(unplacedMesh.id);
    if (onMeshPlacedExt) onMeshPlacedExt(unplacedMesh.id, pos);
    setReticleVisible(false);
    latestHitPosition.current = null;
  }, [unplacedMesh, onMeshPlaced, onMeshPlacedExt, bindMeshToAnchor]);

  // ─── Direct-manipulation gestures ───────────────────────────────────────────
  // Two-finger pinch scales the object; two-finger twist rotates its yaw.
  // Works on any placed object regardless of mode, and selects it on touch —
  // far more natural than the old scale +/- buttons and axis-drag modes.

  /** pinchState: 1=begin, 2=move, 3=end. scaleFactor is cumulative from start. */
  const handlePinch = useCallback(
    (mesh: PlacedMesh, pinchState: number, scaleFactor: number) => {
      if (pinchState === 1) {
        pinchBaseScale.current[mesh.id] = mesh.scale[0];
        if (onMeshSelected) onMeshSelected(mesh.id);
      }
      const base = pinchBaseScale.current[mesh.id] ?? mesh.scale[0];
      const next = Math.min(3, Math.max(0.02, base * scaleFactor));
      if (onMeshScaled) onMeshScaled(mesh.id, next);
    },
    [onMeshScaled, onMeshSelected],
  );

  /** rotateState: 1=begin, 2=move, 3=end. rotationFactor is cumulative degrees. */
  const handleRotate = useCallback(
    (mesh: PlacedMesh, rotateState: number, rotationFactor: number) => {
      if (rotateState === 1) {
        rotateBaseYaw.current[mesh.id] = mesh.rotation[1];
        if (onMeshSelected) onMeshSelected(mesh.id);
      }
      const base = rotateBaseYaw.current[mesh.id] ?? mesh.rotation[1];
      const nextYaw = base - rotationFactor;
      if (onMeshRotated) onMeshRotated(mesh.id, [mesh.rotation[0], nextYaw, mesh.rotation[2]]);
    },
    [onMeshRotated, onMeshSelected],
  );

  // ─── Anchor update logic ───────────────────────────────────────────────────

  /**
   * Extract the world position from a ViroReact anchor object.
   * The ViroAnchor type has a `position` field, but the `center` field on plane
   * anchors gives the center of the detected plane extent in the anchor's local
   * coordinate system. For world-space binding we need the anchor's own position
   * (which IS the world-space transform of the anchor origin). We use `position`
   * first, and only fall back to `center` if position is missing/zero.
   */
  const extractAnchorPosition = useCallback((anchor: any): [number, number, number] | null => {
    // 1) anchor.position — this is the world-space position of the anchor origin
    if (anchor.position && Array.isArray(anchor.position) && anchor.position.length >= 3) {
      const p = anchor.position;
      if (typeof p[0] === 'number' && typeof p[1] === 'number' && typeof p[2] === 'number') {
        return [p[0], p[1], p[2]];
      }
    }

    // 2) anchor.center — plane center (may be in anchor-local or world coords)
    if (anchor.center && Array.isArray(anchor.center) && anchor.center.length >= 3) {
      const c = anchor.center;
      if (typeof c[0] === 'number' && typeof c[1] === 'number' && typeof c[2] === 'number') {
        if (Math.abs(c[0]) > 0.0001 || Math.abs(c[1]) > 0.0001 || Math.abs(c[2]) > 0.0001) {
          return [c[0], c[1], c[2]];
        }
      }
    }

    // 3) anchor.transform.position
    if (anchor.transform?.position && Array.isArray(anchor.transform.position)) {
      const tp = anchor.transform.position;
      if (typeof tp[0] === 'number' && typeof tp[1] === 'number' && typeof tp[2] === 'number') {
        return [tp[0], tp[1], tp[2]];
      }
    }

    return null;
  }, []);

  /**
   * Apply a found/updated plane anchor: record it, (re)bind nearby placed
   * meshes, and slide bound meshes by the anchor's refined transform so they
   * stay glued to the real surface.
   */
  const applyAnchor = useCallback(
    (anchor: any) => {
      if (!anchor?.anchorId) return;
      const type = String(anchor.type ?? 'plane').toLowerCase();
      if (!type.includes('plane')) return; // only bind to plane anchors

      const position = extractAnchorPosition(anchor);
      if (!position) return; // no usable position data

      arAnchorsRef.current[anchor.anchorId] = { position };

      let changed: Record<string, [number, number, number]> | null = null;
      for (const mesh of meshesRef.current) {
        if (!mesh.isPlaced) continue;
        let binding = meshBindingRef.current[mesh.id];

        // Bind a placed-but-unbound mesh (e.g. from a loaded layout) once a
        // suitable anchor appears near it.
        if (!binding) {
          const cur = positionsRef.current[mesh.id];
          if (cur && findNearestAnchor(cur) === anchor.anchorId) {
            meshBindingRef.current[mesh.id] = {
              anchorId: anchor.anchorId,
              offset: [cur[0] - position[0], cur[1] - position[1], cur[2] - position[2]],
            };
            binding = meshBindingRef.current[mesh.id];
          }
        }

        if (binding && binding.anchorId === anchor.anchorId) {
          const np: [number, number, number] = [
            position[0] + binding.offset[0],
            position[1] + binding.offset[1],
            position[2] + binding.offset[2],
          ];
          const cur = positionsRef.current[mesh.id];
          const moved =
            !cur ||
            Math.abs(cur[0] - np[0]) > 0.005 ||
            Math.abs(cur[1] - np[1]) > 0.005 ||
            Math.abs(cur[2] - np[2]) > 0.005;
          if (moved) {
            changed = changed ?? {};
            changed[mesh.id] = np;
          }
        }
      }

      if (changed) {
        const delta = changed;
        setPositions((prev) => ({ ...prev, ...delta }));
        for (const [id, p] of Object.entries(delta)) {
          positionsRef.current[id] = p;
          if (onMeshMoved) onMeshMoved(id, p);
        }
      }
    },
    [findNearestAnchor, onMeshMoved, extractAnchorPosition],
  );

  const handleAnchorRemoved = useCallback((anchor: any) => {
    if (!anchor?.anchorId) return;
    delete arAnchorsRef.current[anchor.anchorId];
    for (const [meshId, b] of Object.entries(meshBindingRef.current)) {
      if (b.anchorId === anchor.anchorId) delete meshBindingRef.current[meshId];
    }
  }, []);

  return (
    <ViroARScene
      onTrackingUpdated={handleTrackingUpdated}
      onCameraARHitTest={handleCameraARHitTest}
      onClick={handleSceneClick}
      onAnchorFound={(anchor: any) => {
        console.log('[ARMeshScene] onAnchorFound:', JSON.stringify({
          id: anchor?.anchorId,
          type: anchor?.type,
          center: anchor?.center,
          position: anchor?.position,
          keys: anchor ? Object.keys(anchor) : [],
        }));
        applyAnchor(anchor);
      }}
      onAnchorUpdated={(anchor: any) => applyAnchor(anchor)}
      onAnchorRemoved={(anchor: any) => handleAnchorRemoved(anchor)}
      onCameraTransformUpdate={(e: any) => {
        if (onCameraPoseUpdate) onCameraPoseUpdate(e.cameraTransform);
      }}
      anchorDetectionTypes={['PlanesHorizontal', 'PlanesVertical']}
    >
      {/* Balanced ambient fill — low enough that the directional light can
          create real form/highlights on PBR GLB materials (flat 1000-intensity
          ambient washed everything out and made meshes look worse than on PC). */}
      <ViroAmbientLight color="#FFFFFF" intensity={350} />
      {/* Key light from above-front, casting soft shadows so objects read as
          grounded instead of floating. */}
      <ViroDirectionalLight
        color="#FFFFFF"
        direction={[0.2, -1, -0.3]}
        intensity={900}
        castsShadow={true}
        shadowOrthographicPosition={[0, 3, 0]}
        shadowOrthographicSize={5} // Tighten ortho size (5m instead of 12m) to concentrate resolution
        shadowNearZ={0.1}
        shadowFarZ={8} // Lower clip distance (8m instead of 16m) to skip far meshes
        shadowOpacity={0.50}
        shadowMapSize={512} // 512x512 resolution (4x cheaper than default 1024) to cool GPU
      />

      {hasUnplacedMesh && <ARReticle position={reticlePosition} visible={reticleVisible} />}

      {/* Dynamic spotlight tracking the active object to highlight it - only rendered in select mode
          to completely eliminate per-pixel light calculation during active drags, rotations, and scales */}
      {activeMeshId && interactionMode === 'select' && positions[activeMeshId] && (
        <ViroSpotLight
          position={[
            positions[activeMeshId][0],
            positions[activeMeshId][1] + 2.5, // 2.5 meters above the object
            positions[activeMeshId][2],
          ]}
          direction={[0, -1, 0]} // Pointing straight down
          color="#00D2FF" // Vibrant cyan / neon blue spotlight
          intensity={1000}
          innerAngle={5}
          outerAngle={20}
          attenuationStartDistance={1.0}
          attenuationEndDistance={4.0}
        />
      )}

      {meshes.map((mesh) => {
        if (!mesh.isPlaced || !mesh.meshSource) return null;

        const isActive = mesh.id === activeMeshId;
        const meshPos = positions[mesh.id] ?? props.initialPositions?.[mesh.id] ?? [0, 0, -1];
        const planeAnchor = planeAnchors[mesh.id] ?? props.initialPositions?.[mesh.id] ?? meshPos;
        const enableDrag =
          isActive && (interactionMode === 'move-floor' || interactionMode === 'move-lift');

        const isCalibrated = calibratedMeshes[mesh.id] === true;
        const modelOffset = modelOffsets[mesh.id] ?? [0, 0, 0];

        return (
          <React.Fragment key={mesh.id}>
            {/* Invisible floor that only renders the shadow cast on it, so the
                object looks anchored to the real surface. Sits at the plane
                anchor where the mesh was placed. */}
            <ViroQuad
              position={[planeAnchor[0], planeAnchor[1], planeAnchor[2]]}
              rotation={[-90, 0, 0]}
              width={3}
              height={3}
              arShadowReceiver={true}
            />
            {isActive && (
              <ViroQuad
                position={[meshPos[0], planeAnchor[1] + 0.002, meshPos[2]]}
                rotation={[-90, mesh.rotation[1], 0]}
                width={1.2 * mesh.scale[0]}
                height={1.2 * mesh.scale[2]}
                materials={['selectionRing']}
                animation={{
                  name: 'selectionPulseLoop',
                  run: true,
                  loop: true,
                }}
              />
            )}
            <ViroNode
              key={`${mesh.id}-${enableDrag}-${interactionMode}`}
              position={meshPos}
              rotation={isCalibrated ? mesh.rotation : [0, 0, 0]}
              scale={isCalibrated ? mesh.scale : [1, 1, 1]}
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
                      positionsRef.current[mesh.id] = newPos;

                      // Also update planeAnchor so the shadow receiver and selection ring follow the mesh
                      setPlaneAnchors((prev) => ({
                        ...prev,
                        [mesh.id]: newPos,
                      }));

                      // Re-anchor at the dropped location so it stays glued there.
                      const b = meshBindingRef.current[mesh.id];
                      const a = b ? arAnchorsRef.current[b.anchorId] : undefined;
                      if (b && a) {
                        b.offset = [
                          newPos[0] - a.position[0],
                          newPos[1] - a.position[1],
                          newPos[2] - a.position[2],
                        ];
                      } else {
                        bindMeshToAnchor(mesh.id, newPos);
                      }
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
              onPinch={(pinchState: number, scaleFactor: number) =>
                handlePinch(mesh, pinchState, scaleFactor)
              }
              onRotate={(rotateState: number, rotationFactor: number) =>
                handleRotate(mesh, rotateState, rotationFactor)
              }
            >
              <ViroNode
                position={modelOffset}
                rotation={[180, 0, 0]} // Rotate 180 degrees around X-axis to flip it right-side up
                scale={[1, 1, 1]}
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
                  onError={(e: any) => {
                    console.warn('Mesh load error:', e.nativeEvent?.error, mesh.meshName);
                    if (props.onMeshLoadError) {
                      props.onMeshLoadError(mesh.id, mesh.meshName);
                    }
                  }}
                />
              </ViroNode>
            </ViroNode>

            {/* Calibration clone: rendered at the world origin [0,0,0] with no rotation or scale,
                almost fully transparent (opacity 0.01) so it's loaded by the engine but invisible to the user.
                Once calibrated, this node is unmounted. */}
            {!isCalibrated && (
              <ViroNode
                position={[0, 0, 0]}
                rotation={[0, 0, 0]}
                scale={[1, 1, 1]}
                opacity={0.01}
              >
                <Viro3DObject
                  ref={(ref) => {
                    if (ref) {
                      calibrationRefs.current[mesh.id] = ref;
                    } else {
                      delete calibrationRefs.current[mesh.id];
                    }
                  }}
                  source={mesh.meshSource}
                  type={mesh.meshType}
                  position={[0, 0, 0]}
                  scale={[1, 1, 1]}
                  onLoadEnd={async () => {
                    if (!calibratedMeshes[mesh.id]) {
                      const ref = calibrationRefs.current[mesh.id];
                      if (ref) {
                        try {
                          // Wait a brief moment for engine physics/bounding thread alignment
                          await new Promise((resolve) => setTimeout(resolve, 150));
                          const box = await ref.getBoundingBoxAsync();
                          if (box && box.boundingBox) {
                            const { minX, maxX, minY, maxY, minZ, maxZ } = box.boundingBox;
                            
                            // Since clone is at [0,0,0], world coordinates are exactly local coordinates!
                            const centerX = (minX + maxX) / 2;
                            const centerZ = (minZ + maxZ) / 2;

                            // Since we rotate 180 degrees around X to flip the upside-down model right-side up,
                            // the local Y-coordinates are negated. The top Y of the raw model (maxY) becomes
                            // the new bottom Y. So we shift by +maxY to align the new bottom with Y = 0.
                            const offsetX = -centerX;
                            const offsetY = maxY;
                            const offsetZ = centerZ;

                            console.log(`[Calibration Clone] Calibrated and Flipped ${mesh.meshName} (${mesh.id}):`, {
                              minX, maxX, minY, maxY, minZ, maxZ,
                              offset: [offsetX, offsetY, offsetZ],
                            });

                            setModelOffsets((prev) => ({
                              ...prev,
                              [mesh.id]: [offsetX, offsetY, offsetZ],
                            }));
                            setCalibratedMeshes((prev) => ({
                              ...prev,
                              [mesh.id]: true,
                            }));
                          }
                        } catch (err) {
                          console.warn('[Calibration Clone] Failed to get bounding box:', err);
                        }
                      }
                    }
                  }}
                />
              </ViroNode>
            )}
          </React.Fragment>
        );
      })}
    </ViroARScene>
  );
}
