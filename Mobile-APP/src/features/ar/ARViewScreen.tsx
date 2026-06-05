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
import { saveLayout, RoomLayout } from '../../services/mesh/layoutStorage';

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
  const [initialPositions, setInitialPositions] = useState<Record<string, [number, number, number]>>({});

  const meshInfo: MeshInfo | null = useMemo(() => {
    if (!meshId || !meshName || !meshType || !meshUri) return null;
    return {
      id: meshId,
      name: meshName,
      format: meshType,
      size: 0,
      uri: meshUri,
      isBundled: !!isBundled,
    };
  }, [meshId, meshName, meshType, meshUri, isBundled]);

  // Build initial PlacedMesh synchronously from route params
  const initialMesh = useMemo<PlacedMesh | null>(() => {
    if (!meshInfo) return null;
    const config = getMeshSource(meshInfo);
    return {
      id: meshId!,
      meshInfo,
      meshSource: config.source,
      meshType: meshType!,
      meshName: meshName!,
      rotation: [0, 0, 0],
      scale: [0.2, 0.2, 0.2],
      isPlaced: false,
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meshInfo]); // only once on mount

  const [meshes, setMeshes] = useState<PlacedMesh[]>(initialMesh ? [initialMesh] : []);
  const [activeMeshId, setActiveMeshId] = useState<string | null>(initialMesh ? meshId! : null);

  // Per-mesh rotation accumulation — keyed by mesh id, avoids re-creating PanResponder
  const meshRotationRefs = useRef<Record<string, [number, number, number]>>(
    initialMesh ? { [meshId!]: [0, 0, 0] } : {}
  );

  // Track positions to save layout without forcing re-renders
  const meshPositionsRef = useRef<Record<string, [number, number, number]>>({});

  // Layouts
  // Ghost Image Alignment
  const [aligningLayout, setAligningLayout] = useState<RoomLayout | null>(null);

  // Derived state
  const hasUnplacedMesh = meshes.some((m) => !m.isPlaced);
  const anyMeshPlaced = meshes.some((m) => m.isPlaced);
  const activeMesh = meshes.find((m) => m.id === activeMeshId) ?? null;

  // Load available meshes for the "Add Object" picker
  useEffect(() => {
    getAvailableMeshes()
      .then(setAvailableMeshes)
      .catch(() => {});
      
    // If a layout was passed via route params, load it on mount
    if (route.params.layout) {
      handleLoadLayout(route.params.layout);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
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
    if (!initialMesh) {
      setMeshes([]);
      setActiveMeshId(null);
      meshRotationRefs.current = {};
      meshPositionsRef.current = {};
      setInitialPositions({});
      setInteractionMode('select');
      setReticleVisible(false);
      setResetCounter((c) => c + 1);
      return;
    }
    // Reset to just the original mesh, unplaced
    meshRotationRefs.current = { [initialMesh.id]: [0, 0, 0] };
    meshPositionsRef.current = {};
    setInitialPositions({});
    const freshMesh: PlacedMesh = {
      ...initialMesh,
      rotation: [0, 0, 0],
      scale: [0.2, 0.2, 0.2],
      isPlaced: false,
    };
    setMeshes([freshMesh]);
    setActiveMeshId(initialMesh.id);
    setInteractionMode('place');
    setReticleVisible(false);
    setResetCounter((c) => c + 1);
  }, [initialMesh]);

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

  const handleMeshPlacedExt = useCallback((id: string, position: [number, number, number]) => {
    meshPositionsRef.current[id] = position;
  }, []);

  const handleMeshMoved = useCallback((id: string, position: [number, number, number]) => {
    meshPositionsRef.current[id] = position;
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
      meshInfo: prepared,
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

  const handleSaveLayout = useCallback(async (name: string) => {
    const layoutMeshes = meshes.filter(m => m.isPlaced).map(m => ({
      meshInfo: m.meshInfo,
      position: meshPositionsRef.current[m.id] ?? [0, 0, -1],
      rotation: meshRotationRefs.current[m.id] ?? [0, 0, 0],
      scale: m.scale,
    }));

    if (layoutMeshes.length === 0) {
      Alert.alert('No meshes placed', 'Place at least one mesh to save a layout.');
      return;
    }

    let screenshotUri: string | undefined;
    
    // Briefly hide UI to take a clean screenshot of the room
    setInteractionMode('select');
    
    try {
      const nav = arNavigatorRef.current as any;
      if (nav && nav._takeScreenshot) {
        // give scene a tiny moment to hide reticle/UI
        await new Promise(r => setTimeout(r, 100));
        const res = await nav._takeScreenshot(`layout_${Date.now()}`, false);
        if (res?.success && res.url) {
          // ViroReact returns a direct file:// URL to the cached screenshot
          // Using it directly avoids ExponentFileSystem.moveAsync rejections
          // Ensure it starts with file:// for the Image component
          screenshotUri = res.url.startsWith('file://') ? res.url : `file://${res.url}`;
        }
      }
    } catch (e) {
      console.warn('Ghost screenshot failed:', e);
    }

    const layout: RoomLayout = {
      id: `layout_${Date.now()}`,
      name: name || `Layout ${new Date().toLocaleTimeString()}`,
      createdAt: Date.now(),
      meshes: layoutMeshes,
      screenshotUri,
    };

    try {
      await saveLayout(layout);
      if (screenshotUri) {
        Alert.alert('Saved', 'Room layout and ghost image saved successfully.');
      } else {
        Alert.alert('Saved', 'Room layout saved, but ghost image capture failed.');
      }
    } catch (e) {
      Alert.alert('Error', 'Failed to save room layout.');
    }
  }, [meshes]);

  const executeLoadLayout = useCallback(async (layout: RoomLayout) => {
    try {
      const newMeshes: PlacedMesh[] = [];
      const newPositions: Record<string, [number, number, number]> = {};
      
      meshRotationRefs.current = {};
      meshPositionsRef.current = {};

      for (let i = 0; i < layout.meshes.length; i++) {
        const item = layout.meshes[i];
        if (!item) continue;
        let prepared = item.meshInfo;
        try {
          prepared = await prepareMeshForViro(item.meshInfo);
        } catch (e) {
          console.warn('prepareMeshForViro failed during load, using local URI:', e);
        }

        const config = getMeshSource(prepared);
        const newId = `${prepared.id}-${i}`;
        
        newMeshes.push({
          id: newId,
          meshInfo: prepared,
          meshSource: config.source,
          meshType: prepared.format as 'GLB' | 'OBJ',
          meshName: prepared.name,
          rotation: item.rotation,
          scale: item.scale,
          isPlaced: true,
        });

        meshRotationRefs.current[newId] = item.rotation;
        meshPositionsRef.current[newId] = item.position;
        newPositions[newId] = item.position;
      }

      setInitialPositions(newPositions);
      setMeshes(newMeshes);
      setActiveMeshId(newMeshes[0]?.id ?? null);
      setInteractionMode('select');
      setResetCounter((c) => c + 1); // Not really needed if we restart AR session, but good to keep
    } catch (e) {
      console.warn('Failed to load layout:', e);
      Alert.alert('Error', 'Could not load the saved layout.');
    }
  }, []);

  const handleLoadLayout = useCallback((layout: RoomLayout) => {
    if (layout.screenshotUri) {
      setAligningLayout(layout);
    } else {
      // Legacy layouts without a screenshot
      executeLoadLayout(layout);
    }
  }, [executeLoadLayout]);

  const confirmAlignment = useCallback(() => {
    if (aligningLayout) {
      // Instead of completely unmounting the Navigator (which crashes ViroReact),
      // we reset the AR session tracking. This clears anchors and makes [0,0,0] the current camera position.
      const nav = arNavigatorRef.current as any;
      if (nav && nav.resetARSession) {
        nav.resetARSession(true, true);
      }
      
      executeLoadLayout(aligningLayout);
      setAligningLayout(null);
    }
  }, [aligningLayout, executeLoadLayout]);

  const cancelAlignment = useCallback(() => {
    setAligningLayout(null);
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
          initialPositions,
          onTrackingChanged: handleTrackingChanged,
          onMeshPlaced: handleMeshPlaced,
          onMeshPlacedExt: handleMeshPlacedExt,
          onMeshMoved: handleMeshMoved,
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
        activeMeshName={activeMesh?.meshName ?? meshName ?? 'AR Scene'}
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
        onSaveLayout={handleSaveLayout}
        onLoadLayout={handleLoadLayout}
        aligningLayout={aligningLayout ?? undefined}
        onConfirmAlignment={confirmAlignment}
        onCancelAlignment={cancelAlignment}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
