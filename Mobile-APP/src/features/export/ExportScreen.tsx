import React, { useCallback, useState } from 'react';
import { Alert, Platform, StyleSheet, Text, TextInput, View } from 'react-native';
import { useCameraDevice } from 'react-native-vision-camera';
import { useTheme } from '../../shared/theme';
import { Header, Card, Button } from '../../shared/components';
import { useSession } from '../../providers/SessionProvider';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';
import { exportMetricBundle } from './services/exportMetricBundle';

type Props = NativeStackScreenProps<RootStackParamList, 'Export'>;

export default function ExportScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const device = useCameraDevice('back');
  const { keyframes, getMetadata } = useSession();
  const metadata = getMetadata();
  const [sceneId, setSceneId] = useState('room_scan');
  const [exportPath, setExportPath] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = useCallback(async () => {
    try {
      setIsExporting(true);
      const result = await exportMetricBundle(metadata, {
        sceneId: sceneId.trim() || 'room_scan',
        cameraDiagonalFovDeg: device?.formats[0]?.fieldOfView ?? 60,
        appVersion: '1.0.0',
        deviceModel: device?.name ?? Platform.OS,
      });
      const visiblePath = result.publicBundleUri ?? result.bundleRoot;
      setExportPath(visiblePath);
      const successMessage = result.publicBundleUri
        ? `Saved ${result.frameCount} ARCore frames and metadata to selected folder:\n${result.publicBundleUri}\n\nPrivate backup:\n${result.bundleRoot}`
        : `Saved ${result.frameCount} ARCore frames and metadata to app storage:\n${result.bundleRoot}\n\nTip: On Android, grant folder access when prompted to export to a visible folder.`;
      Alert.alert('Export complete', successMessage);
    } catch (error) {
      console.error(error);
      Alert.alert('Export failed', 'The Module 1 bundle could not be written.');
    } finally {
      setIsExporting(false);
    }
  }, [device, metadata, sceneId]);

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Header title="Export" onBack={() => navigation.goBack()} />

      <View style={{ padding: theme.spacing.lg, gap: theme.spacing.lg }}>
        <Card elevated>
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.h4.fontSize,
              fontWeight: theme.typography.h4.fontWeight,
              marginBottom: theme.spacing.md,
            }}
          >
            Session Summary
          </Text>

          <View style={styles.row}>
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.body.fontSize,
              }}
            >
              Keyframes
            </Text>
            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              {keyframes.length}
            </Text>
          </View>

          <View style={styles.row}>
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.body.fontSize,
              }}
            >
              Started
            </Text>
            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              {metadata.startedAt ? new Date(metadata.startedAt).toLocaleTimeString() : '-'}
            </Text>
          </View>

          <View style={styles.row}>
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.body.fontSize,
              }}
            >
              Status
            </Text>
            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
            >
              {metadata.captureStatus}
            </Text>
          </View>
        </Card>

        <Card elevated>
          <Text
            style={{
              color: theme.colors.textPrimary,
              fontSize: theme.typography.body.fontSize,
              fontWeight: '700',
              marginBottom: theme.spacing.sm,
            }}
          >
            Scene ID
          </Text>
          <TextInput
            value={sceneId}
            onChangeText={setSceneId}
            placeholder="living_room_a"
            placeholderTextColor={theme.colors.textSecondary}
            style={[
              styles.input,
              {
                color: theme.colors.textPrimary,
                borderColor: theme.colors.border,
                backgroundColor: theme.colors.card,
              },
            ]}
            autoCapitalize="none"
          />
          {exportPath ? (
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.caption.fontSize,
                marginTop: theme.spacing.md,
              }}
            >
              {exportPath}
            </Text>
          ) : null}
        </Card>

        <Button
          title={isExporting ? 'Exporting...' : 'Export Module 1 Bundle'}
          onPress={() => void handleExport()}
          variant="primary"
          size="lg"
          disabled={keyframes.length === 0 || isExporting}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
});
