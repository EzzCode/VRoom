import React, { useEffect, useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { useTheme } from '../../shared/theme';
import { Header, Card, Button, ProgressBar } from '../../shared/components';
import { useSession } from '../../providers/SessionProvider';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';
import {
  JobState,
  JobStatus,
  getJobResultUrl,
  pollJobUntilComplete,
} from './uploadService';
import { deleteSessionFiles } from './sessionPackager';

type Props = NativeStackScreenProps<RootStackParamList, 'ReconstructionStatus'>;

const STAGE_ORDER: JobState[] = ['queued', 'colmap', 'gaussian-training', 'mesh-extraction', 'done'];

const STAGE_LABEL: Record<JobState, string> = {
  queued: 'Queued',
  colmap: 'Running COLMAP (structure from motion)',
  'gaussian-training': 'Training Gaussian splats',
  'mesh-extraction': 'Extracting mesh',
  done: 'Done',
  failed: 'Failed',
};

function stageProgress(state: JobState, withinStage: number): number {
  if (state === 'failed') return 0;
  if (state === 'done') return 1;
  const idx = STAGE_ORDER.indexOf(state);
  if (idx < 0) return 0;
  // Each stage spans 1/(N-1) of the total bar (queued..done).
  const total = STAGE_ORDER.length - 1;
  return Math.min(1, (idx + withinStage) / total);
}

async function downloadResult(jobId: string, sessionId: string): Promise<string> {
  const dest = `${FileSystem.documentDirectory}meshes/reconstructed_${sessionId}.glb`;
  await FileSystem.makeDirectoryAsync(`${FileSystem.documentDirectory}meshes/`, {
    intermediates: true,
  });
  const result = await FileSystem.downloadAsync(getJobResultUrl(jobId), dest);
  return result.uri;
}

export default function ReconstructionStatusScreen({ navigation, route }: Props) {
  const { theme } = useTheme();
  const { jobId } = route.params;
  const { keyframes } = useSession();

  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;

    (async () => {
      try {
        const final = await pollJobUntilComplete(jobId, setStatus, controller.signal);
        if (final.state === 'failed') {
          setError(final.message ?? 'Reconstruction failed.');
          return;
        }
        setDownloading(true);
        const localUri = await downloadResult(jobId, final.jobId);
        await deleteSessionFiles(keyframes);
        navigation.replace('ARView', {
          meshId: `reconstructed_${final.jobId}`,
          meshName: 'New scan',
          meshUri: localUri,
          meshType: 'GLB',
          isBundled: false,
        });
      } catch (e) {
        if (!controller.signal.aborted) {
          setError((e as Error).message);
        }
      } finally {
        setDownloading(false);
      }
    })();

    return () => {
      controller.abort();
    };
  }, [jobId, keyframes, navigation]);

  const handleCancel = () => {
    Alert.alert('Cancel reconstruction?', 'You can still return to your session.', [
      { text: 'Keep waiting', style: 'cancel' },
      {
        text: 'Cancel',
        style: 'destructive',
        onPress: () => {
          abortRef.current?.abort();
          navigation.goBack();
        },
      },
    ]);
  };

  const currentState: JobState = status?.state ?? 'queued';
  const progress = stageProgress(currentState, status?.progress ?? 0);

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Header title="Reconstructing…" onBack={handleCancel} />

      <View style={{ padding: theme.spacing.lg, gap: theme.spacing.lg }}>
        <Card elevated>
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.h4.fontSize,
              fontWeight: theme.typography.h4.fontWeight,
              marginBottom: theme.spacing.sm,
            }}
          >
            {error ? 'Failed' : downloading ? 'Downloading mesh…' : STAGE_LABEL[currentState]}
          </Text>
          {!error && (
            <ProgressBar
              progress={downloading ? 1 : progress}
              color={theme.colors.primary}
            />
          )}
          {status?.message && (
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.mono.fontSize,
                marginTop: theme.spacing.sm,
              }}
            >
              {status.message}
            </Text>
          )}
          {error && (
            <Text
              style={{
                color: theme.colors.error,
                fontSize: theme.typography.body.fontSize,
                marginTop: theme.spacing.sm,
              }}
            >
              {error}
            </Text>
          )}
        </Card>

        <Card elevated>
          {STAGE_ORDER.map((stage) => {
            const idx = STAGE_ORDER.indexOf(currentState);
            const stageIdx = STAGE_ORDER.indexOf(stage);
            const active = stageIdx === idx;
            const done = stageIdx < idx || currentState === 'done';
            return (
              <Text
                key={stage}
                style={{
                  color: done
                    ? theme.colors.success
                    : active
                      ? theme.colors.primary
                      : theme.colors.textSecondary,
                  fontSize: theme.typography.body.fontSize,
                  marginBottom: 6,
                }}
              >
                {done ? '✓ ' : active ? '▸ ' : '· '}
                {STAGE_LABEL[stage]}
              </Text>
            );
          })}
        </Card>

        {error && (
          <Button title="Back to session" onPress={() => navigation.goBack()} variant="primary" size="lg" />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
});
