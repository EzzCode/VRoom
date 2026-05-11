import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../../shared/theme';
import { Header, Card, Button } from '../../shared/components';
import { useSession } from '../../providers/SessionProvider';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Export'>;

export default function ExportScreen({ navigation }: Props) {
  const { theme } = useTheme();
  const { keyframes, getMetadata } = useSession();
  const metadata = getMetadata();

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
              {metadata.startedAt ? new Date(metadata.startedAt).toLocaleTimeString() : '—'}
            </Text>
          </View>
        </Card>

        <Button
          title="Export Keyframes"
          onPress={() => {}}
          variant="primary"
          size="lg"
          disabled={keyframes.length === 0}
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
});
