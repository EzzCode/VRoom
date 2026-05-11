import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { Vec3 } from '../shared/core/types';

type InteractionMode = 'place' | 'move' | 'rotate' | 'scale';
type TrackingState = 'unavailable' | 'limited' | 'normal';

interface ARContextValue {
  selectedMeshUri: string | null;
  setSelectedMeshUri: (uri: string | null) => void;
  selectedMeshType: 'GLB' | 'OBJ';
  setSelectedMeshType: (t: 'GLB' | 'OBJ') => void;
  interactionMode: InteractionMode;
  setInteractionMode: (mode: InteractionMode) => void;
  trackingState: TrackingState;
  setTrackingState: (state: TrackingState) => void;
  isMeshPlaced: boolean;
  setIsMeshPlaced: (placed: boolean) => void;
  isMeshLoading: boolean;
  setIsMeshLoading: (loading: boolean) => void;
  meshPosition: Vec3;
  setMeshPosition: (pos: Vec3) => void;
  meshRotation: Vec3;
  setMeshRotation: (rot: Vec3) => void;
  meshScale: Vec3;
  setMeshScale: (s: Vec3) => void;
  resetPlacement: () => void;
}

const ARContext = createContext<ARContextValue | null>(null);

const DEFAULT_POSITION: Vec3 = [0, 0, -1];
const DEFAULT_ROTATION: Vec3 = [0, 0, 0];
const DEFAULT_SCALE: Vec3 = [0.2, 0.2, 0.2];

export function ARProvider({
  children,
  meshUri,
  meshType,
}: {
  children: React.ReactNode;
  meshUri: string;
  meshType: 'GLB' | 'OBJ';
}) {
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('place');
  const [trackingState, setTrackingState] = useState<TrackingState>('unavailable');
  const [isMeshPlaced, setIsMeshPlaced] = useState(false);
  const [isMeshLoading, setIsMeshLoading] = useState(false);
  const [meshPosition, setMeshPosition] = useState<Vec3>(DEFAULT_POSITION);
  const [meshRotation, setMeshRotation] = useState<Vec3>(DEFAULT_ROTATION);
  const [meshScale, setMeshScale] = useState<Vec3>(DEFAULT_SCALE);

  const resetPlacement = useCallback(() => {
    setIsMeshPlaced(false);
    setMeshPosition(DEFAULT_POSITION);
    setMeshRotation(DEFAULT_ROTATION);
    setMeshScale(DEFAULT_SCALE);
    setInteractionMode('place');
  }, []);

  const value = useMemo<ARContextValue>(
    () => ({
      selectedMeshUri: meshUri,
      setSelectedMeshUri: () => {},
      selectedMeshType: meshType,
      setSelectedMeshType: () => {},
      interactionMode,
      setInteractionMode,
      trackingState,
      setTrackingState,
      isMeshPlaced,
      setIsMeshPlaced,
      isMeshLoading,
      setIsMeshLoading,
      meshPosition,
      setMeshPosition,
      meshRotation,
      setMeshRotation,
      meshScale,
      setMeshScale,
      resetPlacement,
    }),
    [
      meshUri,
      meshType,
      interactionMode,
      trackingState,
      isMeshPlaced,
      isMeshLoading,
      meshPosition,
      meshRotation,
      meshScale,
      resetPlacement,
    ],
  );

  return <ARContext.Provider value={value}>{children}</ARContext.Provider>;
}

export function useAR(): ARContextValue {
  const ctx = useContext(ARContext);
  if (!ctx) {
    throw new Error('useAR must be used within an <ARProvider>');
  }
  return ctx;
}
