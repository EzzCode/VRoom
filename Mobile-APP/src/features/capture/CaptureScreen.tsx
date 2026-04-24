import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Image, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { ViroARSceneNavigator } from '@reactvision/react-viro';
import { useCameraDevice } from 'react-native-vision-camera';
import { CAPTURE_CONFIG } from './config/captureConfig';
import { saveCapturedAsset } from './services/captureStorage';
import { useSession } from '../../providers/SessionProvider';
import { useTheme } from '../../shared/theme';
import { Header, Button, ProgressBar } from '../../shared/components';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';
import ARCaptureScene from './ARCaptureScene';
import { CameraPose, TrackingState } from '../../shared/core/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Capture'>;

export default function CaptureScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const device = useCameraDevice('back');
  const arNavigatorRef = useRef<any>(null);
  const latestPoseRef = useRef<CameraPose | null>(null);
  const latestTrackingStateRef = useRef<TrackingState>('unavailable');
  const isRecordingRef = useRef(false);
  const captureCounterRef = useRef(0);

  const [trackingState, setTrackingState] = useState<TrackingState>('unavailable');
  const [guidance, setGuidance] = useState<string | null>(null);
  const [captureQuality, setCaptureQuality] = useState(0);

  const {
    isRecording,
    startSession,
    stopSession,
    keyframes,
    addKeyframe,
    extractor,
    setCurrentPose,
    setCaptureStatus,
  } = useSession();

  const cameraDiagonalFov = useMemo(
    () => device?.formats[0]?.fieldOfView ?? CAPTURE_CONFIG.coverage.cameraFovDeg,
    [device],
  );

  const handlePoseUpdated = useCallback(
    (pose: CameraPose) => {
      latestPoseRef.current = pose;
      latestTrackingStateRef.current = pose.trackingState;
      setCurrentPose(pose);
      setTrackingState(pose.trackingState);
    },
    [setCurrentPose],
  );

  const handleTrackingChanged = useCallback((state: TrackingState) => {
    latestTrackingStateRef.current = state;
    setTrackingState(state);
  }, []);

  const getImageSize = useCallback((uri: string) => {
    return new Promise<{ width: number; height: number }>((resolve, reject) => {
      Image.getSize(
        uri,
        (width, height) => resolve({ width, height }),
        (error) => reject(error),
      );
    });
  }, []);

  const handleCapture = useCallback(async () => {
    const nav = arNavigatorRef.current;
    const currentPose = latestPoseRef.current;
    if (!nav || !isRecordingRef.current || !currentPose) {
      return;
    }

    if (latestTrackingStateRef.current !== 'normal') {
      setGuidance('Wait for stable ARCore tracking before capturing.');
      return;
    }

    const { shouldCapture, results } = extractor.evaluate(currentPose);
    if (!shouldCapture) {
      const failed = results.find((result) => !result.result.passed);
      setGuidance(failed?.result.reason ?? 'Adjust your position.');
      return;
    }

    setGuidance(null);

    try {
      const frameId = `frame_${String(captureCounterRef.current).padStart(5, '0')}`;
      const screenshot = await nav._takeScreenshot(frameId, false);
      if (!screenshot?.success || !screenshot.url) {
        throw new Error('Viro screenshot failed.');
      }
      const savedPath = await saveCapturedAsset(screenshot.url, `${frameId}.jpg`);
      const size = await getImageSize(savedPath);
      const qualityScore = 1;

      addKeyframe({
        frameId,
        imagePath: savedPath,
        pose: currentPose,
        width: size.width,
        height: size.height,
        qualityScore,
        index: keyframes.length,
      });
      captureCounterRef.current += 1;
      setCaptureQuality(qualityScore);
    } catch (error) {
      console.error('Failed to save AR frame:', error);
      setCaptureStatus('interrupted');
      Alert.alert('Capture failed', 'The AR frame could not be saved. Please try again.');
    }
  }, [addKeyframe, extractor, getImageSize, keyframes.length, setCaptureStatus]);

  useEffect(() => {
    if (!isRecording) {
      return;
    }

    const intervalId = setInterval(() => {
      void handleCapture();
    }, 1200);

    return () => {
      clearInterval(intervalId);
    };
  }, [handleCapture, isRecording]);

  if (device == null) {
    return (
      <View style={[styles.center, { backgroundColor: theme.colors.background }]}>
        <Text style={{ color: theme.colors.textSecondary }}>Loading ARCore camera...</Text>
      </View>
    );
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopSession();
      isRecordingRef.current = false;
    } else {
      startSession();
      setCaptureStatus('aborted');
      captureCounterRef.current = 0;
      isRecordingRef.current = true;
    }
  };

  const trackingProgress =
    trackingState === 'normal' ? 1 : trackingState === 'limited' ? 0.5 : 0.1;

  return (
    <View style={styles.container}>
      <ViroARSceneNavigator
        ref={arNavigatorRef}
        autofocus={true}
        shadowsEnabled={false}
        pbrEnabled={false}
        hdrEnabled={false}
        initialScene={{
          scene: ARCaptureScene as any,
        }}
        viroAppProps={{
          onTrackingChanged: handleTrackingChanged,
          onCameraPose: ({
            position,
            rotation,
            forward,
            up,
            cameraToWorld,
          }: Omit<CameraPose, 'timestampNs' | 'trackingState'>) => {
            handlePoseUpdated({
              position,
              rotation,
              forward,
              up,
              cameraToWorld,
              timestampNs: Date.now() * 1_000_000,
              trackingState: latestTrackingStateRef.current,
            });
          },
        }}
        style={StyleSheet.absoluteFill}
      />

      <View style={[styles.topBar, { paddingTop: CAPTURE_CONFIG.hudTopOffset }]}>
        <Header
          onBack={() => navigation.goBack()}
          transparent
          rightAction={
            keyframes.length > 0 ? (
              <TouchableOpacity onPress={() => navigation.navigate('Export')}>
                <Text
                  style={{
                    color: theme.colors.primary,
                    fontSize: theme.typography.body.fontSize,
                    fontWeight: '600',
                  }}
                >
                  Export
                </Text>
              </TouchableOpacity>
            ) : undefined
          }
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
          <View style={{ flex: 1, marginHorizontal: 12 }}>
            <ProgressBar
              progress={trackingProgress}
              color={
                trackingState === 'normal'
                  ? theme.colors.success
                  : trackingState === 'limited'
                    ? theme.colors.warning
                    : theme.colors.error
              }
            />
          </View>
          <Text
            style={{ color: theme.colors.textSecondary, fontSize: theme.typography.mono.fontSize }}
          >
            {trackingState}
          </Text>
        </View>

        <View
          style={[
            styles.banner,
            {
              backgroundColor: theme.colors.overlay,
              borderRadius: theme.radii.md,
              paddingVertical: theme.spacing.md,
              paddingHorizontal: theme.spacing.xl,
              marginTop: theme.spacing.md,
              borderLeftWidth: 3,
              borderLeftColor: theme.colors.primary,
            },
          ]}
        >
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.body.fontSize,
              fontWeight: '600',
            }}
          >
            {`${Platform.OS} ARCore | FOV ${cameraDiagonalFov.toFixed(1)} deg | Quality ${(captureQuality * 100).toFixed(0)}%`}
          </Text>
        </View>

        {trackingState !== 'normal' && isRecording && (
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
              Move slowly until ARCore tracking is normal.
            </Text>
          </View>
        )}

        {guidance && isRecording && (
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

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
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
  },
});
