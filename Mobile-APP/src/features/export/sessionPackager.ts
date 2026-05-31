import * as FileSystem from 'expo-file-system/legacy';
import { Platform } from 'react-native';
import { Keyframe, SessionMetadata } from '../../shared/core/types';

export interface SessionManifest {
  sessionId: string;
  startedAt: string;
  endedAt: string;
  coveragePercent: number;
  totalFramesAnalysed: number;
  device: {
    platform: string;
    osVersion: string | number;
  };
  keyframes: Array<{
    index: number;
    filename: string;
    pose: Keyframe['pose'];
    blurScore: number;
  }>;
}

function generateSessionId(): string {
  const ts = Date.now().toString(36);
  const rnd = Math.random().toString(36).slice(2, 8);
  return `sess_${ts}_${rnd}`;
}

/**
 * Builds the session manifest (session.json contents) from in-memory
 * session state. Keyframe filenames are derived from the saved imagePath.
 */
export function buildSessionManifest(meta: SessionMetadata): SessionManifest {
  return {
    sessionId: generateSessionId(),
    startedAt: meta.startedAt,
    endedAt: meta.endedAt ?? new Date().toISOString(),
    coveragePercent: meta.coveragePercent,
    totalFramesAnalysed: meta.totalFramesAnalysed,
    device: {
      platform: Platform.OS,
      osVersion: Platform.Version,
    },
    keyframes: meta.keyframes.map((kf) => ({
      index: kf.index,
      filename: (kf.imagePath.split('/').pop() ?? `frame_${kf.index}.jpg`) as string,
      pose: kf.pose,
      blurScore: kf.blurScore,
    })),
  };
}

/** Total bytes of all keyframe images on disk. */
export async function getSessionDiskSize(keyframes: Keyframe[]): Promise<number> {
  let total = 0;
  for (const kf of keyframes) {
    try {
      const info = await FileSystem.getInfoAsync(kf.imagePath);
      total += ((info as { size?: number }).size) ?? 0;
    } catch {
      // ignore missing files
    }
  }
  return total;
}

/** Delete all keyframe JPGs on disk. Call after a successful upload. */
export async function deleteSessionFiles(keyframes: Keyframe[]): Promise<void> {
  await Promise.all(
    keyframes.map((kf) =>
      FileSystem.deleteAsync(kf.imagePath, { idempotent: true }).catch(() => undefined),
    ),
  );
}
