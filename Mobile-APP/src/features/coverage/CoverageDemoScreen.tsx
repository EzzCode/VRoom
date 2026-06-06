// ────────────────────────────────────────────────────────────
// CoverageDemoScreen — exercises the CoverageTracker end-to-end in AR
// ────────────────────────────────────────────────────────────
//
// Lightweight test harness: shows the AR camera feed, drops voxel cubes
// where the user has pointed, and displays live coverage %. Useful for
// validating the pipeline before deciding how to integrate coverage into
// the main CaptureScreen (which currently uses VisionCamera, not Viro).
// ────────────────────────────────────────────────────────────

import React, { useRef, useState, useCallback, useMemo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import CoverageDemoScene from '../../features/coverage/CoverageDemoScene';
import { CoverageTracker, VoxelView } from '../../features/coverage/CoverageTracker';
import { CAPTURE_CONFIG } from '../../features/capture/config/captureConfig';
import { Vec3 } from '../../shared/core/types';
import { useTheme } from '../../shared/theme';
import { Header, ProgressBar } from '../../shared/components';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'CoverageDemo'>;

const VOXEL_CFG = CAPTURE_CONFIG.coverage;

export default function CoverageDemoScreen({ navigation }: Props) {
  const { theme } = useTheme();

  // Local tracker for the demo (independent of SessionProvider).
  const tracker = useMemo(
    () =>
      new CoverageTracker({
        voxelSize: VOXEL_CFG.voxelSize,
        minObservations: VOXEL_CFG.minObservations,
        fovDeg: VOXEL_CFG.cameraFovDeg,
        frustumDepth: VOXEL_CFG.frustumDepth,
      }),
    [],
  );

  const [voxels, setVoxels] = useState<VoxelView[]>([]);
  const [coveragePercent, setCoveragePercent] = useState(0);
  const [tracking, setTracking] = useState<'unavailable' | 'limited' | 'normal'>('unavailable');
  const observeIntervalMs = 200;
  const lastObserveRef = useRef(0);

  // Each hit point is a real surface location along the camera's centre ray.
  // We mark the voxel containing it, so coverage coats geometry the user has
  // actually pointed at — dwelling on a spot fills (greens) it, panning across
  // leaves a trail of partials.
  const handleHitPoint = useCallback(
    (point: Vec3) => {
      const now = Date.now();
      if (now - lastObserveRef.current < observeIntervalMs) return;
      lastObserveRef.current = now;

      tracker.observePoint(point);
      setVoxels(tracker.getVoxels());
      setCoveragePercent(tracker.coveragePercent);
    },
    [tracker],
  );

  const viroAppProps = useMemo(
    () => ({
      voxels,
      voxelSize: VOXEL_CFG.voxelSize,
      onHitPoint: handleHitPoint,
      onTracking: setTracking,
    }),
    [voxels, handleHitPoint],
  );

  return (
    <View style={styles.container}>
      <ViroARSceneNavigator
        autofocus
        initialScene={{ scene: CoverageDemoScene as any }}
        viroAppProps={viroAppProps}
        style={StyleSheet.absoluteFill}
      />

      <View style={[styles.topBar, { paddingTop: 50 }]}>
        <Header onBack={() => navigation.goBack()} transparent />
      </View>

      <View style={[styles.hud, { top: 110 }]}>
        <View
          style={[
            styles.card,
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
            Coverage Demo
          </Text>
          <Text
            style={{
              color: theme.colors.textSecondary,
              fontSize: theme.typography.caption.fontSize,
              marginTop: 4,
            }}
          >
            Tracking: {tracking} · Voxels: {voxels.length}
          </Text>
          <View style={{ marginTop: 8, flexDirection: 'row', alignItems: 'center' }}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <ProgressBar progress={coveragePercent} color={theme.colors.primary} />
            </View>
            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.mono.fontSize,
                fontWeight: '700',
              }}
            >
              {Math.round(coveragePercent * 100)}%
            </Text>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  topBar: { position: 'absolute', left: 0, right: 0 },
  hud: { position: 'absolute', left: 20, right: 20 },
  card: {},
});
