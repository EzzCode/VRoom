import React, { useRef, useCallback, useState, useEffect, useMemo } from 'react';
import { View, StyleSheet, Alert, PanResponder } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import ARMeshScene from './ARMeshScene';
import AROverlayUI from './AROverlayUI';
import { getMeshSource, getAvailableMeshes, prepareMeshForViro } from '../../services/mesh/meshStorage';
import { MeshInfo } from '../../shared/core/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';
import type { PlacedMesh } from './arTypes';

type Props = NativeStackScreenProps<RootStackParamList, 'ARView'>;

export type InteractionMode =
  | 'select'
  | 'place'
  | 'move-floor'
  | 'move-lift'
  | 'rotate-horiz'
  | 'rotate-vert'
  | 'rotate-roll'
  | 'scale';

type TrackingState = 'unavailable' | 'limited' | 'normal';

export default function ARViewScreen({ navigation, route }: Props) {
  const { meshId, meshName, meshUri, meshType, isBundled } = route.params;
  const arNavigatorRef = useRef<any>(null);

  const [interactionMode, setInteractionMode] = useState<InteractionMode>('place');
  const [trackingState, setTrackingState] = useState<TrackingState>('unavailable');
  const [loadingCount, setLoadingCount] = useState(0);
  const [resetCounter, setResetCounter] = useState(0);
  const [reticleVisible, setReticleVisible] = useState(false);
  const [availableMeshes, setAvailableMeshes] = useState<MeshInfo[]>([]);

  // Build initial PlacedMesh synchronously from route params
  const initialMesh = useMemo<PlacedMesh>(() => {
    const config = getMeshSource({
      id: meshId,
      name: meshName,
      format: meshType,
      size: 0,
      uri: meshUri,
      isBundled,
    } as MeshInfo);
    return {
      id: meshId,
      meshSource: config.source,
      meshType,
      meshName,
      rotation: [0, 0, 0],
      scale: [0.2, 0.2, 0.2],
      isPlaced: false,
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // only once on mount

  const [meshes, setMeshes] = useState<PlacedMesh[]>([initialMesh]);
  const [activeMeshId, setActiveMeshId] = useState<string | null>(meshId);

  // Per-mesh rotation accumulation — keyed by mesh id, avoids re-creating PanResponder
  const meshRotationRefs = useRef<Record<string, [number, number, number]>>({
    [meshId]: [0, 0, 0],
  });

  // Derived state
  const hasUnplacedMesh = meshes.some((m) => !m.isPlaced);
  const anyMeshPlaced = meshes.some((m) => m.isPlaced);
  const activeMesh = meshes.find((m) => m.id === activeMeshId) ?? null;

  // Load available meshes for the "Add Object" picker
  useEffect(() => {
    getAvailableMeshes()
      .then(setAvailableMeshes)
      .catch(() => {});
  }, []);

  // Refs so stable PanResponder closures can read fresh values
  const lastPanX = useRef(0);
  const lastPanY = useRef(0);
  const interactionModeRef = useRef(interactionMode);
  interactionModeRef.current = interactionMode;
  const activeMeshIdRef = useRef(activeMeshId);
  activeMeshIdRef.current = activeMeshId;

  const isRotateMode =
    interactionMode === 'rotate-horiz' ||
    interactionMode === 'rotate-vert' ||
    interactionMode === 'rotate-roll';

  const rotatePanResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => {
        const m = interactionModeRef.current;
        return (
          !!activeMeshIdRef.current &&
          (m === 'rotate-horiz' || m === 'rotate-vert' || m === 'rotate-roll')
        );
      },
      onMoveShouldSetPanResponder: () => {
        const m = interactionModeRef.current;
        return (
          !!activeMeshIdRef.current &&
          (m === 'rotate-horiz' || m === 'rotate-vert' || m === 'rotate-roll')
        );
      },
      onPanResponderGrant: () => {
        lastPanX.current = 0;
        lastPanY.current = 0;
      },
      onPanResponderMove: (_, gs) => {
        const dx = gs.dx - lastPanX.current;
        const dy = gs.dy - lastPanY.current;
        lastPanX.current = gs.dx;
        lastPanY.current = gs.dy;
        const SENSITIVITY = 0.4;
        const m = interactionModeRef.current;
        const id = activeMeshIdRef.current;
        if (!id) return;
        if (!meshRotationRefs.current[id]) meshRotationRefs.current[id] = [0, 0, 0];
        const rot = meshRotationRefs.current[id];
        if (m === 'rotate-horiz') rot[1] += dx * SENSITIVITY;
        else if (m === 'rotate-vert') rot[0] += dy * SENSITIVITY;
        else if (m === 'rotate-roll') rot[2] += dx * SENSITIVITY;
        const newRot: [number, number, number] = [rot[0], rot[1], rot[2]];
        setMeshes((prev) =>
          prev.map((mesh) => (mesh.id === id ? { ...mesh, rotation: newRot } : mesh)),
        );
      },
      onPanResponderRelease: () => {
        lastPanX.current = 0;
        lastPanY.current = 0;
      },
    }),
  ).current;

  // ─── Callbacks ────────────────────────────────────────────────────────────

  const handleScreenshot = useCallback(async () => {
    try {
      const nav = arNavigatorRef.current;
      if (nav && nav._takeScreenshot) {
        const result = await nav._takeScreenshot('vroom_ar', true);
        if (result?.success) {
          Alert.alert('Screenshot saved!', 'Your AR view has been saved to your photos.');
        }
      }
    } catch (e) {
      console.warn('Screenshot failed:', e);
    }
  }, []);

  const handleReset = useCallback(() => {
    // Reset to just the original mesh, unplaced
    meshRotationRefs.current = { [meshId]: [0, 0, 0] };
    const freshMesh: PlacedMesh = {
      ...initialMesh,
      rotation: [0, 0, 0],
      scale: [0.2, 0.2, 0.2],
      isPlaced: false,
    };
    setMeshes([freshMesh]);
    setActiveMeshId(meshId);
    setInteractionMode('place');
    setReticleVisible(false);
    setResetCounter((c) => c + 1);
  }, [initialMesh, meshId]);

  const handleScaleChange = useCallback((scale: number) => {
    const id = activeMeshIdRef.current;
    if (!id) return;
    setMeshes((prev) =>
      prev.map((m) => (m.id === id ? { ...m, scale: [scale, scale, scale] } : m)),
    );
  }, []);

  const handleTrackingChanged = useCallback((state: TrackingState) => {
    setTrackingState(state);
  }, []);

  /** Called by scene when the unplaced mesh is tapped into position */
  const handleMeshPlaced = useCallback((id: string) => {
    setMeshes((prev) => prev.map((m) => (m.id === id ? { ...m, isPlaced: true } : m)));
    setActiveMeshId(id);
    setInteractionMode('move-floor');
    setReticleVisible(false);
  }, []);

  /** Called by scene in Select mode when user taps a placed mesh */
  const handleMeshSelected = useCallback((id: string) => {
    setActiveMeshId(id);
  }, []);

  const handleMeshLoading = useCallback((loading: boolean) => {
    setLoadingCount((prev) => (loading ? prev + 1 : Math.max(0, prev - 1)));
  }, []);

  const handleReticleVisible = useCallback((visible: boolean) => {
    setReticleVisible(visible);
  }, []);

  /** Called from the overlay's mesh picker — adds and starts placing a new object */
  const handleAddMesh = useCallback(async (mesh: MeshInfo) => {
    if (mesh.format === 'PLY') return; // shouldn't reach here, but guard anyway

    let prepared = mesh;
    try {
      prepared = await prepareMeshForViro(mesh);
    } catch (e) {
      console.warn('prepareMeshForViro failed, using local URI:', e);
    }

    const config = getMeshSource(prepared);
    const newId = `${prepared.id}-${Date.now()}`;
    const newMesh: PlacedMesh = {
      id: newId,
      meshSource: config.source,
      meshType: prepared.format as 'GLB' | 'OBJ',
      meshName: prepared.name,
      rotation: [0, 0, 0],
      scale: [0.2, 0.2, 0.2],
      isPlaced: false,
    };
    meshRotationRefs.current[newId] = [0, 0, 0];
    setMeshes((prev) => [...prev, newMesh]);
    setActiveMeshId(newId);
    setInteractionMode('place');
  }, []);

  return (
    <View style={styles.container}>
      <ViroARSceneNavigator
        ref={arNavigatorRef}
        autofocus={true}
        shadowsEnabled={true}
        pbrEnabled={false}
        hdrEnabled={false}
        initialScene={{
          scene: ARMeshScene as any,
        }}
        viroAppProps={{
          meshes,
          activeMeshId,
          interactionMode,
          onTrackingChanged: handleTrackingChanged,
          onMeshPlaced: handleMeshPlaced,
          onMeshSelected: handleMeshSelected,
          onMeshLoading: handleMeshLoading,
          onReticleVisible: handleReticleVisible,
          resetRequested: resetCounter,
        }}
        style={StyleSheet.absoluteFill}
      />

      {/* Transparent overlay: captures 1-finger drags in rotate mode only */}
      <View
        style={StyleSheet.absoluteFill}
        pointerEvents={isRotateMode && !!activeMeshId ? 'box-only' : 'none'}
        {...rotatePanResponder.panHandlers}
      />

      <AROverlayUI
        onBack={() => navigation.goBack()}
        onScreenshot={handleScreenshot}
        onReset={handleReset}
        activeMeshName={activeMesh?.meshName ?? meshName}
        interactionMode={interactionMode}
        setInteractionMode={setInteractionMode}
        trackingState={trackingState}
        anyMeshPlaced={anyMeshPlaced}
        hasUnplacedMesh={hasUnplacedMesh}
        isMeshLoading={loadingCount > 0}
        reticleVisible={reticleVisible}
        currentScale={activeMesh?.scale[0] ?? 0.2}
        onScaleChange={handleScaleChange}
        availableMeshes={availableMeshes}
        onAddMesh={handleAddMesh}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
