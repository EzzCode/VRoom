import { NativeModules, Platform } from 'react-native';

export interface CameraCalibration {
  cameraId: string;
  source: string;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  distortion: number[];
  activeArrayScaleX: number;
  activeArrayScaleY: number;
}

type CameraIntrinsicsNativeModule = {
  getBackCameraCalibration: (targetSize: { width: number; height: number }) => Promise<CameraCalibration>;
};

function getNativeModule(): CameraIntrinsicsNativeModule | null {
  const module = (NativeModules as { CameraIntrinsicsModule?: CameraIntrinsicsNativeModule })
    .CameraIntrinsicsModule;
  return module ?? null;
}

export async function getBackCameraCalibration(
  targetSize: { width: number; height: number },
): Promise<CameraCalibration | null> {
  if (Platform.OS !== 'android') {
    return null;
  }
  const module = getNativeModule();
  if (!module) {
    return null;
  }
  try {
    return await module.getBackCameraCalibration(targetSize);
  } catch (error) {
    console.warn('Falling back from native camera calibration.', error);
    return null;
  }
}
