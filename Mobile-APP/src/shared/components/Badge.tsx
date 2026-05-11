import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../theme';

interface BadgeProps {
  label: string;
  variant?: 'format' | 'count' | 'status';
  color?: string;
}

export default function Badge({ label, variant = 'format', color }: BadgeProps) {
  const { theme } = useTheme();

  const bgByVariant = {
    format: theme.colors.primary,
    count: theme.colors.surfaceLight,
    status: color ?? theme.colors.success,
  }[variant];

  const textColorByVariant = {
    format: theme.colors.onPrimary,
    count: theme.colors.textPrimary,
    status: '#FFFFFF',
  }[variant];

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: color ? `${color}20` : bgByVariant,
          borderRadius: theme.radii.sm,
          paddingHorizontal: theme.spacing.sm,
          paddingVertical: 2,
        },
      ]}
    >
      <Text
        style={[
          styles.text,
          {
            color: color ?? textColorByVariant,
            fontSize: theme.typography.caption.fontSize,
            fontWeight: theme.typography.captionBold.fontWeight,
          },
        ]}
      >
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignSelf: 'flex-start',
  },
  text: {},
});
