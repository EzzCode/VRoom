import React, { useCallback, useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useRunOnJS } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { processBlur } from './gates/BlurGate';
import { CAPTURE_CONFIG } from './config/captureConfig';
import { saveCapturedPhoto } from './services/captureStorage';
import { useSession, SessionProvider } from '../../providers/SessionProvider';
import { useTheme } from '../../shared/theme';
import { Header, Button, ProgressBar } from '../../shared/components';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Capture'>;

function shouldProcessFrame(interval: number): boolean {
  'worklet';

  const workletGlobal = globalThis as typeof globalThis & {
    __blurFrameCount?: number;
  };

  const currentCount = workletGlobal.__blurFrameCount ?? 0;
  workletGlobal.__blurFrameCount = currentCount + 1;

  return currentCount % Math.max(1, interval) === 0;
}

function CaptureScreenInner({ navigation }: Props) {
  const { theme } = useTheme();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const { resize } = useResizePlugin();
  const camera = useRef<Camera>(null);

  const [isBlurry, setIsBlurry] = useState(false);
  const [blurScore, setBlurScore] = useState(0);
  const [guidance, setGuidance] = useState<string | null>(null);

  const captureIntervalMs = 1200;

  const { isRecording, startSession, stopSession, keyframes, addKeyframe, extractor, currentPose, coveragePercent } =
    useSession();

  const isRecordingRef = useRef(false);
  const isBlurryRef = useRef(false);
  const blurScoreRef = useRef(0);

  const handleCapture = useCallback(async () => {
    if (!camera.current || !isRecordingRef.current) return;

    const { shouldCapture, results } = extractor.evaluate(currentPose);

    if (!shouldCapture) {
      const failed = results.find((r) => !r.result.passed);
      setGuidance(failed?.result.reason ?? 'Adjust your position.');
      return;
    }

    setGuidance(null);

    try {
      const photo = await camera.current.takePhoto({ flash: 'off' });
      const savedPath = await saveCapturedPhoto(photo.path);

      addKeyframe({
        imagePath: savedPath,
        pose: currentPose ?? {
          position: [0, 0, 0],
          rotation: [0, 0, 0],
          forward: [0, 0, -1],
          up: [0, 1, 0],
          timestamp: Date.now(),
        },
        blurScore: blurScoreRef.current,
        index: keyframes.length,
      });
    } catch (e) {
      console.error('Failed to save frame:', e);
    }
  }, [extractor, currentPose, addKeyframe, keyframes.length]);

  useEffect(() => {
    if (!isRecording) {
      return;
    }

    const intervalId = setInterval(() => {
      if (!isBlurryRef.current) {
        void handleCapture();
      }
    }, captureIntervalMs);

    return () => {
      clearInterval(intervalId);
    };
  }, [isRecording, handleCapture]);

  const updateBlurOnJS = useRunOnJS((blurry: boolean, score: number) => {
    isBlurryRef.current = blurry;
    blurScoreRef.current = score;
    setIsBlurry(blurry);
    setBlurScore(score);
  }, []);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      if (!shouldProcessFrame(CAPTURE_CONFIG.frameSamplingInterval)) {
        return;
      }

      let resizedBuffer: Uint8Array;
      try {
        resizedBuffer = resize(frame, {
          scale: {
            width: CAPTURE_CONFIG.resize.width,
            height: CAPTURE_CONFIG.resize.height,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });
      } catch {
        return;
      }

      const blurResult = processBlur(
        resizedBuffer,
        CAPTURE_CONFIG.resize.width,
        CAPTURE_CONFIG.resize.height,
        3,
      );

      void updateBlurOnJS(blurResult.isBlurry, blurResult.variance);
    },
    [updateBlurOnJS, resize],
  );

  if (!hasPermission) {
    return (
      <View style={[styles.center, { backgroundColor: theme.colors.background }]}>
        <Text
          style={{
            color: theme.colors.textSecondary,
            fontSize: theme.typography.body.fontSize,
            textAlign: 'center',
            marginBottom: 20,
          }}
        >
          VRoom needs camera access to scan.
        </Text>
        <Button title="Grant Permission" onPress={requestPermission} variant="primary" />
      </View>
    );
  }

  if (device == null)
    return (
      <View style={[styles.center, { backgroundColor: theme.colors.background }]}>
        <Text style={{ color: theme.colors.textSecondary }}>Loading camera…</Text>
      </View>
    );

  const toggleRecording = () => {
    if (isRecording) {
      stopSession();
      isRecordingRef.current = false;
    } else {
      startSession();
      isRecordingRef.current = true;
    }
  };

  const blurProgress = Math.min(blurScore / 500, 1);

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
        pixelFormat="yuv"
        frameProcessor={frameProcessor}
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
              progress={blurProgress}
              color={isBlurry ? theme.colors.error : theme.colors.success}
            />
          </View>
          <Text
            style={{ color: theme.colors.textSecondary, fontSize: theme.typography.mono.fontSize }}
          >
            {blurScore.toFixed(0)}
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
            style={{ color: theme.colors.textSecondary, fontSize: theme.typography.mono.fontSize }}
          >
            {Math.round(coveragePercent * 100)}%
          </Text>
        </View>

        {isBlurry && isRecording && (
          <View
            style={[
              styles.banner,
              {
                backgroundColor: theme.colors.errorBackground,
                borderRadius: theme.radii.md,
                paddingVertical: theme.spacing.md,
                paddingHorizontal: theme.spacing.xl,
                marginTop: theme.spacing.md,
                borderLeftWidth: 3,
                borderLeftColor: theme.colors.error,
              },
            ]}
          >
            <Text
              style={{
                color: theme.colors.error,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              Hold Steady! Image too blurry.
            </Text>
          </View>
        )}

        {guidance && isRecording && !isBlurry && (
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

export default function CaptureScreen(props: Props) {
  return (
    <SessionProvider>
      <CaptureScreenInner {...props} />
    </SessionProvider>
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
