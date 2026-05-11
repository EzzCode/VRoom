// ────────────────────────────────────────────────────────────
// Session Provider — Singleton context for capture state
// ────────────────────────────────────────────────────────────
import React, { createContext, useContext, useRef, useState, useCallback, useMemo } from 'react';
import { Keyframe, CameraPose, SessionMetadata } from '../shared/core/types';
import { KeyframeExtractor } from '../features/capture/KeyframeExtractor';
import { AngleGate } from '../features/capture/gates/AngleGate';

// ── Context value type ──────────────────────────────────────
export interface SessionContextValue {
  /** Whether we are actively recording */
  isRecording: boolean;
  /** Start a new recording session */
  startSession: () => void;
  /** Stop the current session */
  stopSession: () => void;
  /** All saved keyframes in the current session */
  keyframes: Keyframe[];
  /** Add a keyframe to the session */
  addKeyframe: (kf: Keyframe) => void;
  /** The keyframe extractor with all registered gates */
  extractor: KeyframeExtractor;
  /** Latest camera pose from AR (null if unavailable) */
  currentPose: CameraPose | null;
  /** Update the current camera pose */
  setCurrentPose: (pose: CameraPose) => void;
  /** Session metadata for export */
  getMetadata: () => SessionMetadata;
  /** Mark the session export lifecycle status */
  setCaptureStatus: (status: SessionMetadata['captureStatus']) => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

// ── Provider component ──────────────────────────────────────
export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [isRecording, setIsRecording] = useState(false);
  const [keyframes, setKeyframes] = useState<Keyframe[]>([]);
  const [currentPose, setCurrentPose] = useState<CameraPose | null>(null);
  const sessionStartRef = useRef<string>('');
  const captureIdRef = useRef<string>('');
  const captureStatusRef = useRef<SessionMetadata['captureStatus']>('aborted');

  // Create the extractor once and register gates
  const extractor = useMemo(() => {
    const ext = new KeyframeExtractor();
    ext.addGate(new AngleGate());
    // BlurGate runs inside the worklet, not in the extractor pipeline.
    // CoverageGate will be added in Build 3.
    return ext;
  }, []);

  const startSession = useCallback(() => {
    setKeyframes([]);
    setCurrentPose(null);
    extractor.resetAll();
    sessionStartRef.current = new Date().toISOString();
    captureIdRef.current = `capture_${Date.now()}`;
    captureStatusRef.current = 'aborted';
    setIsRecording(true);
  }, [extractor]);

  const stopSession = useCallback(() => {
    captureStatusRef.current = 'completed';
    setIsRecording(false);
  }, []);

  const addKeyframe = useCallback((kf: Keyframe) => {
    setKeyframes((prev) => [...prev, kf]);
  }, []);

  const getMetadata = useCallback((): SessionMetadata => {
    return {
      captureId: captureIdRef.current,
      startedAt: sessionStartRef.current,
      endedAt: new Date().toISOString(),
      keyframes,
      captureStatus: captureStatusRef.current,
      coveragePercent: 0, // Populated in Build 3
      totalFramesAnalysed: 0, // Updated by frame processor
    };
  }, [keyframes]);

  const setCaptureStatus = useCallback((status: SessionMetadata['captureStatus']) => {
    captureStatusRef.current = status;
  }, []);

  const value: SessionContextValue = {
    isRecording,
    startSession,
    stopSession,
    keyframes,
    addKeyframe,
    extractor,
    currentPose,
    setCurrentPose,
    getMetadata,
    setCaptureStatus,
  };

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
}

// ── Hook ────────────────────────────────────────────────────
export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error('useSession must be used within a <SessionProvider>');
  }
  return ctx;
}
