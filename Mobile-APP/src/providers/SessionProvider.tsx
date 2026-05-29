// ────────────────────────────────────────────────────────────
// Session Provider — Singleton context for capture state
// ────────────────────────────────────────────────────────────
import React, { createContext, useContext, useRef, useState, useCallback, useMemo } from 'react';
import { Keyframe, CameraPose, SessionMetadata } from '../shared/core/types';
import { KeyframeExtractor } from '../features/capture/KeyframeExtractor';
import { AngleGate } from '../features/capture/gates/AngleGate';
import { CoverageGate } from '../features/capture/gates/CoverageGate';
import { CoverageTracker } from '../features/coverage/CoverageTracker';
import { CAPTURE_CONFIG } from '../features/capture/config/captureConfig';

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
  /** Shared coverage tracker (HUD + gate observe the same instance) */
  coverageTracker: CoverageTracker;
  /** Coverage percent in [0,1], updated when a keyframe is saved */
  coveragePercent: number;
  /** Session metadata for export */
  getMetadata: () => SessionMetadata;
}

const SessionContext = createContext<SessionContextValue | null>(null);

// ── Provider component ──────────────────────────────────────
export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [isRecording, setIsRecording] = useState(false);
  const [keyframes, setKeyframes] = useState<Keyframe[]>([]);
  const [currentPose, setCurrentPose] = useState<CameraPose | null>(null);
  const [coveragePercent, setCoveragePercent] = useState(0);
  const sessionStartRef = useRef<string>('');

  // Shared tracker — CoverageGate peeks, addKeyframe commits, HUD reads.
  const coverageTracker = useMemo(
    () =>
      new CoverageTracker({
        voxelSize: CAPTURE_CONFIG.coverage.voxelSize,
        minObservations: CAPTURE_CONFIG.coverage.minObservations,
        fovDeg: CAPTURE_CONFIG.coverage.cameraFovDeg,
        frustumDepth: CAPTURE_CONFIG.coverage.frustumDepth,
      }),
    [],
  );

  // Create the extractor once and register gates
  const extractor = useMemo(() => {
    const ext = new KeyframeExtractor();
    ext.addGate(new AngleGate());
    ext.addGate(new CoverageGate(coverageTracker));
    // BlurGate runs inside the worklet, not in the extractor pipeline.
    return ext;
  }, [coverageTracker]);

  const startSession = useCallback(() => {
    setKeyframes([]);
    setCurrentPose(null);
    setCoveragePercent(0);
    coverageTracker.reset();
    extractor.resetAll();
    sessionStartRef.current = new Date().toISOString();
    setIsRecording(true);
  }, [extractor, coverageTracker]);

  const stopSession = useCallback(() => {
    setIsRecording(false);
  }, []);

  const addKeyframe = useCallback(
    (kf: Keyframe) => {
      setKeyframes((prev) => [...prev, kf]);
      // Commit this pose's voxel observations now that the frame is saved.
      coverageTracker.observe(kf.pose);
      setCoveragePercent(coverageTracker.coveragePercent);
    },
    [coverageTracker],
  );

  const getMetadata = useCallback((): SessionMetadata => {
    return {
      startedAt: sessionStartRef.current,
      endedAt: new Date().toISOString(),
      keyframes,
      coveragePercent: coverageTracker.coveragePercent,
      totalFramesAnalysed: 0, // Updated by frame processor
    };
  }, [keyframes, coverageTracker]);

  const value: SessionContextValue = {
    isRecording,
    startSession,
    stopSession,
    keyframes,
    addKeyframe,
    extractor,
    currentPose,
    setCurrentPose,
    coverageTracker,
    coveragePercent,
    getMetadata,
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
