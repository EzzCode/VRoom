import React, { useRef, useCallback, useState } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import ARMeshScene from './ARMeshScene';
import AROverlayUI from './AROverlayUI';
import { getMeshSource } from '../../services/mesh/meshStorage';
import { MeshInfo } from '../../shared/core/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'ARView'>;

type InteractionMode = 'place' | 'move' | 'rotate' | 'scale';
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
  }, []);

  const handleTrackingChanged = useCallback((state: TrackingState) => {
    setTrackingState(state);
  }, []);

  const handleMeshPlaced = useCallback((placed: boolean) => {
    setIsMeshPlaced(placed);
    if (placed) {
      setInteractionMode('move');
      setReticleVisible(false);
    }
  }, []);

  const handleMeshLoading = useCallback((loading: boolean) => {
    setIsMeshLoading(loading);
  }, []);

  const handleReticleVisible = useCallback((visible: boolean) => {
    setReticleVisible(visible);
  }, []);

  return (
    <View style={styles.container}>
      <ViroARSceneNavigator
        ref={arNavigatorRef}
        autofocus={true}
        shadowsEnabled={true}
        pbrEnabled={true}
        hdrEnabled={true}
        initialScene={{
          scene: ARMeshScene as any,
        }}
        viroAppProps={{
          meshSource: actualMeshSource,
          meshType,
          meshName,
          interactionMode,
          onTrackingChanged: handleTrackingChanged,
          onMeshPlaced: handleMeshPlaced,
          onMeshLoading: handleMeshLoading,
          onReticleVisible: handleReticleVisible,
          resetRequested: resetCounter,
        }}
        style={StyleSheet.absoluteFill}
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
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
