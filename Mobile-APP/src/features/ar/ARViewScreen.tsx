import React, { useRef, useCallback, useState } from 'react';
import { View, StyleSheet, Alert, PanResponder } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import ARMeshScene from './ARMeshScene';
import AROverlayUI from './AROverlayUI';
import { getMeshSource } from '../../services/mesh/meshStorage';
import { MeshInfo } from '../../shared/core/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'ARView'>;

type InteractionMode = 'place' | 'move-floor' | 'move-lift' | 'rotate-horiz' | 'rotate-vert' | 'rotate-roll' | 'scale';
type TrackingState = 'unavailable' | 'limited' | 'normal';

export default function ARViewScreen({ navigation, route }: Props) {
  const { meshName, meshUri, meshType, isBundled } = route.params;
  const arNavigatorRef = useRef<any>(null);

  const [interactionMode, setInteractionMode] = useState<InteractionMode>('place');
  const [trackingState, setTrackingState] = useState<TrackingState>('unavailable');
  const [isMeshPlaced, setIsMeshPlaced] = useState(false);
  const [isMeshLoading, setIsMeshLoading] = useState(false);
  const [resetCounter, setResetCounter] = useState(0);
  const [reticleVisible, setReticleVisible] = useState(false);
  const [meshScale, setMeshScale] = useState<[number, number, number]>([0.2, 0.2, 0.2]);

  // Single-finger rotate pan state — rotation is owned here, not in the Viro scene
  const rotXRef = useRef(0);
  const rotYRef = useRef(0);
  const rotZRef = useRef(0);
  const [meshRotation, setMeshRotation] = useState<[number, number, number]>([0, 0, 0]);
  const lastPanX = useRef(0);
  const lastPanY = useRef(0);
  // Refs so the stable PanResponder can read fresh values without recreation
  const interactionModeRef = useRef(interactionMode);
  interactionModeRef.current = interactionMode;
  const isMeshPlacedRef = useRef(isMeshPlaced);
  isMeshPlacedRef.current = isMeshPlaced;

  const rotatePanResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => {
        const m = interactionModeRef.current;
        return isMeshPlacedRef.current && (m === 'rotate-horiz' || m === 'rotate-vert' || m === 'rotate-roll');
      },
      onMoveShouldSetPanResponder: () => {
        const m = interactionModeRef.current;
        return isMeshPlacedRef.current && (m === 'rotate-horiz' || m === 'rotate-vert' || m === 'rotate-roll');
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
        if (m === 'rotate-horiz') {
          rotYRef.current += dx * SENSITIVITY;
        } else if (m === 'rotate-vert') {
          rotXRef.current += dy * SENSITIVITY;
        } else if (m === 'rotate-roll') {
          rotZRef.current += dx * SENSITIVITY;
        }
        setMeshRotation([rotXRef.current, rotYRef.current, rotZRef.current]);
      },
      onPanResponderRelease: () => {
        lastPanX.current = 0;
        lastPanY.current = 0;
      },
    }),
  ).current;

  const meshConfig = getMeshSource({
    id: '',
    name: meshName,
    format: meshType,
    size: 0,
    uri: meshUri,
    isBundled,
  } as MeshInfo);

  const actualMeshSource = meshConfig.source;

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
    setIsMeshPlaced(false);
    setInteractionMode('place');
    setReticleVisible(false);
    setResetCounter((c) => c + 1);
    // Reset rotation accumulators
    rotXRef.current = 0;
    rotYRef.current = 0;
    rotZRef.current = 0;
    setMeshRotation([0, 0, 0]);
    setMeshScale([0.2, 0.2, 0.2]);
  }, []);

  const handleScaleChange = useCallback((scale: number) => {
    setMeshScale([scale, scale, scale]);
  }, []);

  const handleTrackingChanged = useCallback((state: TrackingState) => {
    setTrackingState(state);
  }, []);

  const handleMeshPlaced = useCallback((placed: boolean) => {
    setIsMeshPlaced(placed);
    if (placed) {
      setInteractionMode('move-floor');
      setReticleVisible(false);
    }
  }, []);

  const handleMeshLoading = useCallback((loading: boolean) => {
    setIsMeshLoading(loading);
  }, []);

  const handleReticleVisible = useCallback((visible: boolean) => {
    setReticleVisible(visible);
  }, []);

  const isRotateMode =
    interactionMode === 'rotate-horiz' ||
    interactionMode === 'rotate-vert' ||
    interactionMode === 'rotate-roll';

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
          meshSource: actualMeshSource,
          meshType,
          meshName,
          interactionMode,
          meshRotation,
          meshScale,
          onTrackingChanged: handleTrackingChanged,
          onMeshPlaced: handleMeshPlaced,
          onMeshLoading: handleMeshLoading,
          onReticleVisible: handleReticleVisible,
          resetRequested: resetCounter,
        }}
        style={StyleSheet.absoluteFill}
      />

      {/* Transparent overlay: captures 1-finger drags in rotate mode, invisible otherwise */}
      <View
        style={StyleSheet.absoluteFill}
        pointerEvents={isRotateMode && isMeshPlaced ? 'box-only' : 'none'}
        {...rotatePanResponder.panHandlers}
      />

      <AROverlayUI
        onBack={() => navigation.goBack()}
        onScreenshot={handleScreenshot}
        onReset={handleReset}
        meshName={meshName}
        interactionMode={interactionMode}
        setInteractionMode={setInteractionMode}
        trackingState={trackingState}
        isMeshPlaced={isMeshPlaced}
        isMeshLoading={isMeshLoading}
        reticleVisible={reticleVisible}
        currentScale={meshScale[0]}
        onScaleChange={handleScaleChange}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
