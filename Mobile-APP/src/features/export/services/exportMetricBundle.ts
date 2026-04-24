import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { SessionMetadata } from '../../../shared/core/types';
import { estimateIntrinsicsFromVerticalFov } from '../../../shared/core/arPose';

type SafPermissionResult = {
  granted: boolean;
  directoryUri?: string;
};

type StorageAccessFramework = {
  requestDirectoryPermissionsAsync: () => Promise<SafPermissionResult>;
  makeDirectoryAsync: (parentUri: string, dirName: string) => Promise<string>;
  createFileAsync: (parentUri: string, fileName: string, mimeType: string) => Promise<string>;
};

function getDocumentDirectory(): string {
  const documentDirectory = (FileSystem as { documentDirectory?: string }).documentDirectory;
  if (!documentDirectory) {
    throw new Error('Unable to resolve document directory for exports.');
  }
  return documentDirectory;
}

function toRelativeFramesPath(bundleRoot: string, absoluteImagePath: string): string {
  const normalizedRoot = bundleRoot.replace(/\\/g, '/');
  const normalizedImagePath = absoluteImagePath.replace(/\\/g, '/');
  return normalizedImagePath.replace(normalizedRoot, '').replace(/^\/+/, '');
}

async function ensureDir(path: string): Promise<void> {
  await FileSystem.makeDirectoryAsync(path, { intermediates: true });
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await FileSystem.writeAsStringAsync(path, JSON.stringify(value, null, 2));
}

function getStorageAccessFramework(): StorageAccessFramework | null {
  const saf = (FileSystem as { StorageAccessFramework?: StorageAccessFramework }).StorageAccessFramework;
  return saf ?? null;
}

async function writeSafTextFile(
  saf: StorageAccessFramework,
  parentUri: string,
  fileName: string,
  content: string,
  mimeType: string,
): Promise<void> {
  const fileUri = await saf.createFileAsync(parentUri, fileName, mimeType);
  await FileSystem.writeAsStringAsync(fileUri, content);
}

async function copyImageToSaf(
  saf: StorageAccessFramework,
  sourcePath: string,
  parentUri: string,
  fileName: string,
): Promise<void> {
  const base64Data = await FileSystem.readAsStringAsync(sourcePath, {
    encoding: FileSystem.EncodingType.Base64,
  });
  const destinationUri = await saf.createFileAsync(parentUri, fileName, 'image/jpeg');
  await FileSystem.writeAsStringAsync(destinationUri, base64Data, {
    encoding: FileSystem.EncodingType.Base64,
  });
}

export interface ExportMetricBundleOptions {
  sceneId: string;
  cameraDiagonalFovDeg: number;
  appVersion: string;
  deviceModel: string;
}

export interface ExportMetricBundleResult {
  bundleRoot: string;
  manifestPath: string;
  frameCount: number;
  publicBundleUri?: string;
}

export async function exportMetricBundle(
  metadata: SessionMetadata,
  options: ExportMetricBundleOptions,
): Promise<ExportMetricBundleResult> {
  if (metadata.keyframes.length === 0) {
    throw new Error('No keyframes are available to export.');
  }

  const bundleRoot = `${getDocumentDirectory()}exports/${metadata.captureId}/`;
  const framesRoot = `${bundleRoot}frames/`;

  await ensureDir(bundleRoot);
  await ensureDir(framesRoot);

  const intrinsics = metadata.keyframes.map((frame) => {
    const estimated = estimateIntrinsicsFromVerticalFov(
      frame.width,
      frame.height,
      options.cameraDiagonalFovDeg,
    );
    return {
      frame_id: frame.frameId,
      timestamp_ns: frame.pose.timestampNs,
      fx: estimated.fx,
      fy: estimated.fy,
      cx: estimated.cx,
      cy: estimated.cy,
      width: frame.width,
      height: frame.height,
      distortion: [],
    };
  });

  const poses = metadata.keyframes.map((frame) => ({
    frame_id: frame.frameId,
    timestamp_ns: frame.pose.timestampNs,
    tracking_state: frame.pose.trackingState,
    camera_to_world: frame.pose.cameraToWorld,
    pose_source: 'arcore_viro',
  }));

  const tracking = metadata.keyframes.map((frame) => ({
    frame_id: frame.frameId,
    timestamp_ns: frame.pose.timestampNs,
    tracking_state: frame.pose.trackingState,
  }));

  for (const frame of metadata.keyframes) {
    const destination = `${framesRoot}${frame.frameId}.jpg`;
    await FileSystem.copyAsync({
      from: frame.imagePath,
      to: destination,
    });
  }

  const firstFrame = metadata.keyframes[0]!;
  const manifest = {
    scene_id: options.sceneId,
    capture_id: metadata.captureId,
    platform: 'arcore',
    device_model: options.deviceModel,
    app_version: options.appVersion,
    frame_count: metadata.keyframes.length,
    image_width: firstFrame.width,
    image_height: firstFrame.height,
    frame_rate: 1,
    timebase: 'unix_ns',
    world_up_axis: 'y',
    units: 'meters',
    capture_mode: 'arcore_rgb',
    capture_status: metadata.captureStatus,
    frames: metadata.keyframes.map((frame) => ({
      frame_id: frame.frameId,
      path: toRelativeFramesPath(bundleRoot, `${framesRoot}${frame.frameId}.jpg`),
    })),
    platform_detail: Platform.OS,
  };

  await writeJson(`${bundleRoot}manifest.json`, manifest);
  await writeJson(`${bundleRoot}intrinsics.json`, intrinsics);
  await writeJson(`${bundleRoot}poses.json`, poses);
  await writeJson(`${bundleRoot}tracking.json`, tracking);

  let publicBundleUri: string | undefined;
  if (Platform.OS === 'android') {
    const saf = getStorageAccessFramework();
    if (saf) {
      try {
        const permission = await saf.requestDirectoryPermissionsAsync();
        if (permission.granted && permission.directoryUri) {
          let captureDirName = metadata.captureId;
          let captureDirUri: string;
          try {
            captureDirUri = await saf.makeDirectoryAsync(permission.directoryUri, captureDirName);
          } catch {
            captureDirName = `${metadata.captureId}_${Date.now()}`;
            captureDirUri = await saf.makeDirectoryAsync(permission.directoryUri, captureDirName);
          }

          const framesDirUri = await saf.makeDirectoryAsync(captureDirUri, 'frames');

          await writeSafTextFile(
            saf,
            captureDirUri,
            'manifest.json',
            JSON.stringify(manifest, null, 2),
            'application/json',
          );
          await writeSafTextFile(
            saf,
            captureDirUri,
            'intrinsics.json',
            JSON.stringify(intrinsics, null, 2),
            'application/json',
          );
          await writeSafTextFile(
            saf,
            captureDirUri,
            'poses.json',
            JSON.stringify(poses, null, 2),
            'application/json',
          );
          await writeSafTextFile(
            saf,
            captureDirUri,
            'tracking.json',
            JSON.stringify(tracking, null, 2),
            'application/json',
          );

          for (const frame of metadata.keyframes) {
            await copyImageToSaf(
              saf,
              `${framesRoot}${frame.frameId}.jpg`,
              framesDirUri,
              `${frame.frameId}.jpg`,
            );
          }

          publicBundleUri = captureDirUri;
        }
      } catch (error) {
        console.warn('Public export skipped; private export remains available.', error);
      }
    }
  }

  return {
    bundleRoot,
    manifestPath: `${bundleRoot}manifest.json`,
    frameCount: metadata.keyframes.length,
    publicBundleUri,
  };
}
