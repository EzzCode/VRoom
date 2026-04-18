import React from 'react';
import { TouchableOpacity, View, StyleSheet, type ViewStyle } from 'react-native';
import { useTheme } from '../theme';

interface CardProps {
  children: React.ReactNode;
  onPress?: () => void;
  elevated?: boolean;
  style?: ViewStyle;
}

export default function Card({ children, onPress, elevated = false, style }: CardProps) {
  const { theme } = useTheme();

  const containerStyle = [
    styles.card,
    {
      backgroundColor: elevated ? theme.colors.cardElevated : theme.colors.card,
      borderRadius: theme.radii.lg,
      padding: theme.spacing.lg,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    elevated && theme.shadows.md,
    style,
  ];

  if (onPress) {
    return (
      <TouchableOpacity style={containerStyle} onPress={onPress} activeOpacity={0.7}>
        {children}
      </TouchableOpacity>
    );
  }

  return <View style={containerStyle}>{children}</View>;
}

const styles = StyleSheet.create({
  card: {},
});
