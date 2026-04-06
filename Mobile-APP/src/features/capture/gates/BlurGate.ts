// ────────────────────────────────────────────────────────────
// BlurGate — Laplacian variance blur detection (worklet)
//
// Runs inside a VisionCamera frame processor on the worklet
// thread. Uses the official react-native-fast-opencv real-time
// detection pattern: frameBufferToMat → OpenCV pipeline →
// clearBuffers on every frame.
//
// Reference: https://lukaszkurantdev.github.io/react-native-fast-opencv/examples/realtimedetection
// ────────────────────────────────────────────────────────────
import { OpenCV } from 'react-native-fast-opencv';
import { ObjectType, DataTypes, BorderTypes } from 'react-native-fast-opencv';
import { CAPTURE_CONFIG } from '../config/captureConfig';

/**
 * Computes the Laplacian variance of a downscaled camera frame.
 *
 * Pipeline:  BGR buffer → Mat → grayscale → Laplacian → meanStdDev → variance
 *
 * If variance < threshold → blurry.
 *
 * Must be called inside a VisionCamera frame processor worklet.
 *
 * @returns `{ isBlurry, variance }` where variance is the Laplacian variance or 0 on error.
 */
export function processBlur(
  resizedFrame: Uint8Array,
  width: number,
  height: number,
  channels: 1 | 3 = 1,
): { isBlurry: boolean; variance: number } {
  'worklet';

  try {
    const packedLumaSize = width * height;
    const expectedSize = channels === 3 ? packedLumaSize * 3 : packedLumaSize;

    if (resizedFrame.length < expectedSize) {
      return { isBlurry: false, variance: 0 };
    }

    // Build a grayscale plane from packed input.
    let grayPixels: Uint8Array;
    if (channels === 1) {
      grayPixels = resizedFrame.subarray(0, packedLumaSize);
    } else {
      grayPixels = new Uint8Array(packedLumaSize);
      for (let i = 0; i < packedLumaSize; i += 1) {
        const srcIdx = i * 3;
        const r = resizedFrame[srcIdx] ?? 0;
        const g = resizedFrame[srcIdx + 1] ?? 0;
        const b = resizedFrame[srcIdx + 2] ?? 0;
        // Integer luma approximation: 0.299R + 0.587G + 0.114B
        grayPixels[i] = (77 * r + 150 * g + 29 * b) >> 8;
      }
    }

    // 1. Convert the raw Grayscale pixel buffer into an OpenCV Mat natively.
    // Notice channels = 1 (CV_8U).
    const grayMat = OpenCV.bufferToMat('uint8', height, width, 1, grayPixels);

    // 2. Compute the Laplacian (second-derivative edge map)
    const laplacianMat = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_64F);
    OpenCV.invoke(
      'Laplacian',
      grayMat, // Note: No cvtColor needed, we are already perfectly grayscale natively!
      laplacianMat,
      DataTypes.CV_64F,
      1, // ksize
      1, // scale
      0, // delta
      BorderTypes.BORDER_DEFAULT,
    );

    // 3. Extract Laplacian values and compute variance directly in the worklet.
    // This avoids relying on meanStdDev output mats, which can read back as zero
    // on some Android builds even when the Laplacian itself is valid.
    const { buffer: laplacianBuffer } = OpenCV.matToBuffer(laplacianMat, 'float64');
    const sampleCount = laplacianBuffer.length;
    if (sampleCount === 0) {
      return { isBlurry: false, variance: 0 };
    }

    let sum = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      sum += laplacianBuffer[i] ?? 0;
    }

    const mean = sum / sampleCount;
    let squaredDiffSum = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      const value = laplacianBuffer[i] ?? 0;
      const delta = value - mean;
      squaredDiffSum += delta * delta;
    }

    const variance = squaredDiffSum / sampleCount;
    const isBlurry = variance < CAPTURE_CONFIG.blurThreshold;
    
    return { isBlurry, variance };
  } catch (e: any) {
    // Fail gracefully: return neutral state if OpenCV fails
    return { isBlurry: false, variance: 0 };
  } finally {
    // CRITICAL: release all native Mats to prevent memory leaks.
    OpenCV.clearBuffers();
  }
}
