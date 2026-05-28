import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useTheme } from '../theme';
import Card from './Card';
import Badge from './Badge';

interface MeshCardProps {
  name: string;
  format: 'GLB' | 'OBJ';
  size: string;
  onPress: () => void;
  onDelete?: () => void;
}

export default function MeshCard({ name, format, size, onPress, onDelete }: MeshCardProps) {
  const { theme } = useTheme();

  return (
    <Card onPress={onPress} style={styles.card}>
      <View
        style={[
          styles.iconContainer,
          {
            backgroundColor: theme.colors.surfaceLight,
            borderRadius: theme.radii.md,
          },
        ]}
      >
        <Ionicons name="cube-outline" size={32} color={theme.colors.primary} />
        {onDelete && (
          <Pressable
            onPress={onDelete}
            hitSlop={8}
            style={({ pressed }) => [
              styles.deleteButton,
              { backgroundColor: theme.colors.error ?? '#E53935', opacity: pressed ? 0.7 : 1 },
            ]}
          >
            <Ionicons name="close" size={16} color="#fff" />
          </Pressable>
        )}
      </View>

      <View style={styles.info}>
        <Text
          style={[
            styles.name,
            {
              color: theme.colors.textPrimary,
              fontSize: theme.typography.body.fontSize,
              fontWeight: theme.typography.bodyBold.fontWeight,
            },
          ]}
          numberOfLines={1}
        >
          {name}
        </Text>

        <View style={styles.meta}>
          <Badge label={format} variant="format" />
          <Text
            style={[
              styles.sizeText,
              {
                color: theme.colors.textTertiary,
                fontSize: theme.typography.caption.fontSize,
              },
            ]}
          >
            {size}
          </Text>
        </View>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    width: '100%',
  },
  iconContainer: {
    width: '100%',
    aspectRatio: 1.4,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  info: {
    gap: 4,
  },
  name: {},
  meta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sizeText: {},
  deleteButton: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
