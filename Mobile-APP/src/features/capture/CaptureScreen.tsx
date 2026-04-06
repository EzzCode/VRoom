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
import { useSession } from '../../providers/SessionProvider';

// ── Helpers ─────────────────────────────────────────────────
function shouldProcessFrame(interval: number): boolean {
  'worklet';

  const workletGlobal = globalThis as typeof globalThis & {
    __blurFrameCount?: number;
  };

  const currentCount = workletGlobal.__blurFrameCount ?? 0;
  workletGlobal.__blurFrameCount = currentCount + 1;

  return currentCount % Math.max(1, interval) === 0;
}

// ── Capture Screen ──────────────────────────────────────────
export default function CaptureScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const { resize } = useResizePlugin();
  const camera = useRef<Camera>(null);

  const [isBlurry, setIsBlurry] = useState(false);
  const [blurScore, setBlurScore] = useState(0);
  const [guidance, setGuidance] = useState<string | null>(null);

  const captureIntervalMs = 1200;

  const {
    isRecording,
    startSession,
    stopSession,
    keyframes,
    addKeyframe,
    extractor,
    currentPose,
  } = useSession();

  // Ref mirrors isRecording for worklet access
  const isRecordingRef = useRef(false);

  const handleCapture = useCallback(async () => {
    if (!camera.current || !isRecordingRef.current) return;

    // ── Run JS-side gates (angle diversity, etc.) ──
    const { shouldCapture, results } = extractor.evaluate(currentPose);

    if (!shouldCapture) {
      // Find the first failed gate's reason for the HUD
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
        blurScore,
        index: keyframes.length,
      });
    } catch (e) {
      console.error('Failed to save frame:', e);
    }
  }, [extractor, currentPose, addKeyframe, keyframes.length, blurScore]);

  useEffect(() => {
    if (!isRecording) {
      return;
    }

    const intervalId = setInterval(() => {
      if (!isBlurry) {
        void handleCapture();
      }
    }, captureIntervalMs);

    return () => {
      clearInterval(intervalId);
    };
  }, [isRecording, isBlurry, handleCapture]);

  const updateBlurOnJS = useRunOnJS(
    (blurry: boolean, score: number) => {
      setIsBlurry(blurry);
      setBlurScore(score);
    },
    [],
  );

  // ── Frame processor (runs on worklet thread) ──────────────
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      if (!shouldProcessFrame(CAPTURE_CONFIG.frameSamplingInterval)) {
        return;
      }

      // 1. Use the VisionCamera resize plugin to obtain a packed RGB buffer.
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
        // Resize plugin unavailable or frame conversion failed; skip this frame.
        return;
      }

      // 2. Run blur detection on the resized RGB frame.
      const blurResult = processBlur(
        resizedBuffer,
        CAPTURE_CONFIG.resize.width,
        CAPTURE_CONFIG.resize.height,
        3,
      );

      // 3. Bridge results back to JS only on sampled frames.
      void updateBlurOnJS(blurResult.isBlurry, blurResult.variance);
    },
    [updateBlurOnJS, resize],
  );

  // ── Permission guard ──────────────────────────────────────
  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>VRoom needs camera access to scan.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (device == null)
    return (
      <View style={styles.center}>
        <Text style={styles.text}>Loading camera…</Text>
      </View>
    );

  // ── Handlers ──────────────────────────────────────────────
  const toggleRecording = () => {
    if (isRecording) {
      stopSession();
      isRecordingRef.current = false;
    } else {
      startSession();
      isRecordingRef.current = true;
    }
  };

  // ── Render ────────────────────────────────────────────────
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

      {/* ── HUD Overlay ── */}
      <View style={styles.hud}>
        <View style={styles.statsBar}>
          <Text style={styles.statText}>
            Keyframes: {keyframes.length}
          </Text>
          <Text style={styles.statText}>
            Blur: {blurScore.toFixed(1)}
          </Text>
        </View>

        {/* Blur warning */}
        {isBlurry && isRecording && (
          <View style={styles.warningBanner}>
            <Text style={styles.warningText}>
              Hold Steady! Image too blurry.
            </Text>
          </View>
        )}

        {/* Gate guidance (angle diversity, coverage, etc.) */}
        {guidance && isRecording && !isBlurry && (
          <View style={styles.guidanceBanner}>
            <Text style={styles.guidanceText}>{guidance}</Text>
          </View>
        )}
      </View>

      {/* ── Record Controls ── */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording && styles.recordingActive,
          ]}
          onPress={toggleRecording}
        >
          <Text style={styles.buttonText}>
            {isRecording ? 'Stop' : 'Start Capture'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Styles ──────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#111',
  },
  hud: {
    position: 'absolute',
    top: CAPTURE_CONFIG.hudTopOffset,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  statsBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    backgroundColor: 'rgba(0,0,0,0.55)',
    padding: 10,
    borderRadius: 10,
  },
  statText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  warningBanner: {
    marginTop: 16,
    backgroundColor: 'rgba(255,50,50,0.9)',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
  },
  warningText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  guidanceBanner: {
    marginTop: 16,
    backgroundColor: 'rgba(255,180,0,0.9)',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
  },
  guidanceText: { color: '#000', fontSize: 16, fontWeight: '600' },
  controls: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
  },
  recordButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 30,
  },
  recordingActive: { backgroundColor: '#FF3B30' },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: '600' },
  text: { fontSize: 18, marginBottom: 20, textAlign: 'center', color: '#ccc' },
  button: { backgroundColor: '#007AFF', padding: 15, borderRadius: 8 },
});
