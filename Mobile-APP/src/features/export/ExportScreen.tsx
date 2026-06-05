import React, { useEffect, useMemo, useState } from 'react';
import { Alert, Image, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../../shared/theme';
import { Header, Card, Button, ProgressBar } from '../../shared/components';
import { useSession } from '../../providers/SessionProvider';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';
import { buildSessionManifest, deleteSessionFiles, getSessionDiskSize } from './sessionPackager';
import { uploadSession } from './uploadService';
import { formatFileSize } from '../../services/mesh/meshStorage';

type Props = NativeStackScreenProps<RootStackParamList, 'Export'>;

function formatDuration(startISO: string, endISO: string): string {
  if (!startISO || !endISO) return '—';
  const ms = new Date(endISO).getTime() - new Date(startISO).getTime();
  if (!isFinite(ms) || ms < 0) return '—';
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
}

export default function ExportScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const { keyframes, getMetadata } = useSession();
  const metadata = useMemo(() => getMetadata(), [getMetadata]);

  const [diskBytes, setDiskBytes] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    let cancelled = false;
    void getSessionDiskSize(keyframes).then((n) => {
      if (!cancelled) setDiskBytes(n);
    });
    return () => {
      cancelled = true;
    };
  }, [keyframes]);

  const handleUpload = async () => {
    if (keyframes.length === 0) return;
    setUploading(true);
    setUploadProgress(0);
    try {
      const manifest = buildSessionManifest(metadata);
      const result = await uploadSession(manifest, keyframes, setUploadProgress);
      navigation.replace('ReconstructionStatus', { jobId: result.jobId });
    } catch (e) {
      Alert.alert('Upload failed', (e as Error).message);
      setUploading(false);
    }
  };

  const handleDiscard = () => {
    if (keyframes.length === 0) {
      navigation.goBack();
      return;
    }
    Alert.alert(
      'Discard session?',
      `${keyframes.length} keyframes (${formatFileSize(diskBytes)}) will be deleted.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: async () => {
            await deleteSessionFiles(keyframes);
            navigation.popToTop();
          },
        },
      ],
    );
  };

  const stat = (label: string, value: string | number) => (
    <View style={styles.row}>
      <Text style={{ color: theme.colors.textSecondary, fontSize: theme.typography.body.fontSize }}>
        {label}
      </Text>
      <Text
        style={{
          color: theme.colors.textPrimary,
          fontSize: theme.typography.body.fontSize,
          fontWeight: '600',
        }}
      >
        {value}
      </Text>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Header title="Session Summary" onBack={() => navigation.goBack()} />

      <ScrollView contentContainerStyle={{ padding: theme.spacing.lg, gap: theme.spacing.lg }}>
        <Card elevated>
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.h4.fontSize,
              fontWeight: theme.typography.h4.fontWeight,
              marginBottom: theme.spacing.md,
            }}
          >
            Overview
          </Text>
          {stat('Keyframes', keyframes.length)}
          {stat('Coverage', `${(metadata.coveragePercent * 100).toFixed(0)}%`)}
          {stat('Duration', formatDuration(metadata.startedAt, metadata.endedAt ?? ''))}
          {stat('On disk', formatFileSize(diskBytes))}
        </Card>

        <Card elevated>
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.h4.fontSize,
              fontWeight: theme.typography.h4.fontWeight,
              marginBottom: theme.spacing.md,
            }}
          >
            Keyframes
          </Text>
          {keyframes.length === 0 ? (
            <Text style={{ color: theme.colors.textSecondary }}>
              No keyframes captured. Try again with more movement.
            </Text>
          ) : (
            <View style={styles.thumbnailGrid}>
              {keyframes.map((kf) => {
                const uri = kf.imagePath.startsWith('file://')
                  ? kf.imagePath
                  : `file://${kf.imagePath}`;
                return (
                  <Image
                    key={kf.index}
                    source={{ uri }}
                    style={[styles.thumb, { borderColor: theme.colors.border }]}
                  />
                );
              })}
            </View>
          )}
        </Card>

        {uploading && (
          <Card elevated>
            <Text style={{ color: theme.colors.textPrimary, marginBottom: theme.spacing.sm }}>
              Uploading… {(uploadProgress * 100).toFixed(0)}%
            </Text>
            <ProgressBar progress={uploadProgress} color={theme.colors.primary} />
          </Card>
        )}

        <Button
          title={uploading ? 'Uploading…' : 'Upload & Reconstruct'}
          onPress={handleUpload}
          variant="primary"
          size="lg"
          disabled={keyframes.length === 0 || uploading}
        />

        <Button
          title="Discard"
          onPress={handleDiscard}
          variant="ghost"
          size="lg"
          disabled={uploading}
        />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  thumbnailGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  thumb: {
    width: 88,
    height: 88,
    borderRadius: 6,
    borderWidth: StyleSheet.hairlineWidth,
  },
});
