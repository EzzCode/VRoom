import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../../shared/theme';
import { Card } from '../../shared/components';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  const { theme } = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.hero, { padding: theme.spacing.xxl }]}>
        <Text
          style={{
            color: theme.colors.primary,
            fontSize: 42,
            fontWeight: '800',
          }}
        >
          VRoom
        </Text>
        <Text
          style={{
            color: theme.colors.textSecondary,
            fontSize: theme.typography.body.fontSize,
            marginTop: 8,
          }}
        >
          Scan, project, and explore rooms in AR
        </Text>
      </View>

      <View style={[styles.actions, { padding: theme.spacing.lg, gap: theme.spacing.lg }]}>
        <Card
          onPress={() => navigation.navigate('Capture')}
          style={{ borderLeftWidth: 4, borderLeftColor: theme.colors.primary }}
        >
          <View style={[styles.actionIconRow, { gap: theme.spacing.md }]}>
            <View
              style={[
                styles.iconCircle,
                { backgroundColor: theme.colors.primary + '20', borderRadius: theme.radii.md },
              ]}
            >
              <Text style={{ fontSize: 24 }}>📷</Text>
            </View>
            <View style={styles.actionText}>
              <Text
                style={{
                  color: theme.colors.textPrimary,
                  fontSize: theme.typography.h4.fontSize,
                  fontWeight: theme.typography.h4.fontWeight,
                }}
              >
                Start New Scan
              </Text>
              <Text
                style={{
                  color: theme.colors.textTertiary,
                  fontSize: theme.typography.caption.fontSize,
                }}
              >
                Capture your room with quality-gated frames
              </Text>
            </View>
          </View>
        </Card>

        <Card
          onPress={() => navigation.navigate('MeshGallery')}
          style={{ borderLeftWidth: 4, borderLeftColor: theme.colors.secondary }}
        >
          <View style={[styles.actionIconRow, { gap: theme.spacing.md }]}>
            <View
              style={[
                styles.iconCircle,
                {
                  backgroundColor: theme.colors.secondary + '20',
                  borderRadius: theme.radii.md,
                },
              ]}
            >
              <Text style={{ fontSize: 24 }}>🧊</Text>
            </View>
            <View style={styles.actionText}>
              <Text
                style={{
                  color: theme.colors.textPrimary,
                  fontSize: theme.typography.h4.fontSize,
                  fontWeight: theme.typography.h4.fontWeight,
                }}
              >
                View in AR
              </Text>
              <Text
                style={{
                  color: theme.colors.textTertiary,
                  fontSize: theme.typography.caption.fontSize,
                }}
              >
                Project 3D meshes into your room
              </Text>
            </View>
          </View>
        </Card>

        <Card
          onPress={() => navigation.navigate('CoverageDemo')}
          style={{ borderLeftWidth: 4, borderLeftColor: theme.colors.warning }}
        >
          <View style={[styles.actionIconRow, { gap: theme.spacing.md }]}>
            <View
              style={[
                styles.iconCircle,
                {
                  backgroundColor: theme.colors.warning + '20',
                  borderRadius: theme.radii.md,
                },
              ]}
            >
              <Text style={{ fontSize: 24 }}>🗺️</Text>
            </View>
            <View style={styles.actionText}>
              <Text
                style={{
                  color: theme.colors.textPrimary,
                  fontSize: theme.typography.h4.fontSize,
                  fontWeight: theme.typography.h4.fontWeight,
                }}
              >
                Coverage Demo
              </Text>
              <Text
                style={{
                  color: theme.colors.textTertiary,
                  fontSize: theme.typography.caption.fontSize,
                }}
              >
                Visualise scan coverage as voxels in AR
              </Text>
            </View>
          </View>
        </Card>

        <Card
          onPress={() => navigation.navigate('SavedLayouts')}
          style={{ borderLeftWidth: 4, borderLeftColor: theme.colors.textPrimary }}
        >
          <View style={[styles.actionIconRow, { gap: theme.spacing.md }]}>
            <View
              style={[
                styles.iconCircle,
                {
                  backgroundColor: theme.colors.textPrimary + '20',
                  borderRadius: theme.radii.md,
                },
              ]}
            >
              <Text style={{ fontSize: 24 }}>🛋️</Text>
            </View>
            <View style={styles.actionText}>
              <Text
                style={{
                  color: theme.colors.textPrimary,
                  fontSize: theme.typography.h4.fontSize,
                  fontWeight: theme.typography.h4.fontWeight,
                }}
              >
                Saved Layouts
              </Text>
              <Text
                style={{
                  color: theme.colors.textTertiary,
                  fontSize: theme.typography.caption.fontSize,
                }}
              >
                Restore a saved room configuration
              </Text>
            </View>
          </View>
        </Card>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  hero: {
    alignItems: 'center',
    paddingTop: 80,
    paddingBottom: 32,
  },
  actions: {},
  actionIconRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconCircle: {
    width: 52,
    height: 52,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionText: {
    flex: 1,
    gap: 2,
  },
});
