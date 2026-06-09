// ────────────────────────────────────────────────────────────
// ARCaptureScreen — AR-based capture with surface coverage
// ────────────────────────────────────────────────────────────
//
// Alternative to the VisionCamera CaptureScreen. Runs on the Viro AR
// camera, so it gets real 6DoF pose (AngleGate gains translation) and
// surface-accurate coverage via AR hit-testing. There is no real-time
// blur gate here (the VisionCamera frame worklet can't run alongside
// Viro) — keyframe gating relies on AngleGate + CoverageGate + the
// richer pose. Keyframes are captured as AR snapshots.
// ────────────────────────────────────────────────────────────

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import ARCaptureScene from './ARCaptureScene';
import { saveCapturedPhoto } from '../services/captureStorage';
import { VoxelView } from '../../coverage/CoverageTracker';
import { CAPTURE_CONFIG } from '../config/captureConfig';
import { useSession } from '../../../providers/SessionProvider';
import { useTheme } from '../../../shared/theme';
import { Header, Button, ProgressBar } from '../../../shared/components';
import { CameraPose, Vec3 } from '../../../shared/core/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'ARCapture'>;

const VOXEL_CFG = CAPTURE_CONFIG.coverage;
const CAPTURE_INTERVAL_MS = 1200;

function ARCaptureScreenInner({ navigation }: Props) {
  const { theme } = useTheme();
  const arNavigatorRef = useRef<any>(null);

  const {
    isRecording,
    startSession,
    stopSession,
    keyframes,
    addKeyframe,
    addCoveragePoint,
    extractor,
    currentPose,
    setCurrentPose,
    coverageTracker,
    coveragePercent,
  } = useSession();

  const [voxels, setVoxels] = useState<VoxelView[]>([]);
  const [tracking, setTracking] = useState<'unavailable' | 'limited' | 'normal'>('unavailable');
  const [guidance, setGuidance] = useState<string | null>(null);
  const [hideVoxels, setHideVoxels] = useState(false);

  const isRecordingRef = useRef(false);
  const currentPoseRef = useRef(currentPose);
  useEffect(() => {
    currentPoseRef.current = currentPose;
  }, [currentPose]);

  // Real 6DoF pose from the AR camera → feeds AngleGate + keyframe metadata.
  const handlePose = useCallback(
    (pose: CameraPose) => {
      setCurrentPose(pose);
    },
    [setCurrentPose],
  );

  // Surface hit point → commit coverage and refresh the overlay.
  const handleHitPoint = useCallback(
    (point: Vec3) => {
      if (!isRecordingRef.current) return;
      addCoveragePoint(point);
      setVoxels(coverageTracker.getVoxels());
    },
    [addCoveragePoint, coverageTracker],
  );

  const handleCapture = useCallback(async () => {
    if (!isRecordingRef.current) return;

    const pose = currentPoseRef.current;
    const { shouldCapture, results } = extractor.evaluate(pose);
    if (!shouldCapture) {
      const failed = results.find((r) => !r.result.passed);
      setGuidance(failed?.result.reason ?? 'Move to a new angle.');
      return;
    }
    setGuidance(null);

    try {
      // Hide voxels temporarily so they don't get baked into the screenshot
      setHideVoxels(true);
      // Wait a short moment for the React render and native AR scene to update
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Viro exposes _takeScreenshot (underscore-prefixed) on the navigator ref.
      // saveToCameraRoll=false keeps it out of the gallery.
      const nav = arNavigatorRef.current;
      if (!nav?._takeScreenshot) {
        setHideVoxels(false);
        return;
      }
      
      const shot = await (nav._takeScreenshot as Function)(`vroom_kf_${Date.now()}`, false);
      
      // Restore voxels immediately after the shot
      setHideVoxels(false);
      
      const sourceUri: string | undefined = shot?.url;
      if (!shot?.success || !sourceUri) return;

      const savedPath = await saveCapturedPhoto(sourceUri, 'arpose');
      addKeyframe(
        {
          imagePath: savedPath,
          pose: pose ?? {
            position: [0, 0, 0],
            rotation: [0, 0, 0],
            forward: [0, 0, -1],
            up: [0, 1, 0],
            timestamp: Date.now(),
          },
          blurScore: 0, // no real-time blur scoring on the AR path
          index: keyframes.length,
        },
        { observeCoverage: false }, // coverage already committed via hit points
      );
    } catch (e) {
      console.error('AR keyframe capture failed:', e);
    }
  }, [extractor, addKeyframe, keyframes.length]);

  useEffect(() => {
    if (!isRecording) return;
    const intervalId = setInterval(() => {
      void handleCapture();
    }, CAPTURE_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [isRecording, handleCapture]);

  const viroAppProps = useMemo(
    () => ({
      voxels: hideVoxels ? [] : voxels,
      voxelSize: VOXEL_CFG.voxelSize,
      onPose: handlePose,
      onHitPoint: handleHitPoint,
      onTracking: setTracking,
    }),
    [voxels, hideVoxels, handlePose, handleHitPoint],
  );

  const toggleRecording = () => {
    if (isRecording) {
      stopSession();
      isRecordingRef.current = false;
      // Defer navigation so stopSession's state flush completes before we
      // read keyframes.length — otherwise the stale count may be 0 and
      // the navigate is skipped (or, if it fires during unmount, crashes).
      setTimeout(() => {
        if (keyframes.length > 0) {
          navigation.navigate('Export');
        }
      }, 0);
    } else {
      setVoxels([]);
      setGuidance(null);
      startSession();
      isRecordingRef.current = true;
    }
  };

  const trackingReady = tracking === 'normal';

  return (
    <View style={styles.container}>
      <ViroARSceneNavigator
        ref={arNavigatorRef}
        autofocus
        initialScene={{ scene: ARCaptureScene as any }}
        viroAppProps={viroAppProps}
        style={StyleSheet.absoluteFill}
      />

      <View style={[styles.topBar, { paddingTop: CAPTURE_CONFIG.hudTopOffset }]}>
        <Header
          onBack={() => navigation.goBack()}
          transparent
        />
      </View>

      <View style={[styles.hud, { top: CAPTURE_CONFIG.hudTopOffset + 56 }]}>
        <View
          style={[
            styles.statsBar,
            {
              backgroundColor: theme.colors.overlay,
              borderRadius: theme.radii.md,
              padding: theme.spacing.md,
            },
          ]}
        >
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.body.fontSize,
              fontWeight: '700',
            }}
          >
            Keyframes: {keyframes.length}
          </Text>
          <Text
            style={{
              color: trackingReady ? theme.colors.success : theme.colors.warning,
              fontSize: theme.typography.mono.fontSize,
              marginLeft: 'auto',
            }}
          >
            {trackingReady ? '● Tracking' : tracking === 'limited' ? '◌ Limited' : '○ No AR'}
          </Text>
        </View>

        <View
          style={[
            styles.statsBar,
            {
              backgroundColor: theme.colors.overlay,
              borderRadius: theme.radii.md,
              padding: theme.spacing.md,
              marginTop: theme.spacing.sm,
            },
          ]}
        >
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.body.fontSize,
              fontWeight: '700',
            }}
          >
            Coverage
          </Text>
          <View style={{ flex: 1, marginHorizontal: 12 }}>
            <ProgressBar progress={coveragePercent} color={theme.colors.primary} />
          </View>
          <Text
            style={{ color: 'rgba(255,255,255,0.7)', fontSize: theme.typography.mono.fontSize }}
          >
            {Math.round(coveragePercent * 100)}%
          </Text>
        </View>

        {isRecording && !trackingReady && (
          <View
            style={[
              styles.banner,
              {
                backgroundColor: theme.colors.warningBackground,
                borderRadius: theme.radii.md,
                paddingVertical: theme.spacing.md,
                paddingHorizontal: theme.spacing.xl,
                marginTop: theme.spacing.md,
                borderLeftWidth: 3,
                borderLeftColor: theme.colors.warning,
              },
            ]}
          >
            <Text
              style={{
                color: theme.colors.warning,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              Move your phone slowly to start AR tracking…
            </Text>
          </View>
        )}

        {guidance && isRecording && trackingReady && (
          <View
            style={[
              styles.banner,
              {
                backgroundColor: theme.colors.warningBackground,
                borderRadius: theme.radii.md,
                paddingVertical: theme.spacing.md,
                paddingHorizontal: theme.spacing.xl,
                marginTop: theme.spacing.md,
                borderLeftWidth: 3,
                borderLeftColor: theme.colors.warning,
              },
            ]}
          >
            <Text
              style={{
                color: theme.colors.warning,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              {guidance}
            </Text>
          </View>
        )}
      </View>

      <View style={styles.controls}>
        {keyframes.length > 0 && !isRecording && (
          <TouchableOpacity
            onPress={() => navigation.navigate('Export')}
            style={[styles.exportButton, { borderColor: theme.colors.primary }]}
          >
            <Text style={{ color: theme.colors.primary, fontSize: theme.typography.body.fontSize, fontWeight: '600' }}>
              Review & Export
            </Text>
          </TouchableOpacity>
        )}
        <Button
          title={isRecording ? 'Stop Capture' : 'Start Capture'}
          onPress={toggleRecording}
          variant={isRecording ? 'danger' : 'primary'}
          size="lg"
        />
      </View>
    </View>
  );
}

export default function ARCaptureScreen(props: Props) {
  return <ARCaptureScreenInner {...props} />;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  topBar: {
    position: 'absolute',
    left: 0,
    right: 0,
  },
  hud: {
    position: 'absolute',
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  statsBar: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
  },
  banner: {
    width: '100%',
  },
  controls: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    alignItems: 'center',
    gap: 12,
  },
  exportButton: {
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 24,
  },
});
