import React from 'react';
import {
  TouchableOpacity,
  Text,
  ActivityIndicator,
  StyleSheet,
  type ViewStyle,
  type TextStyle,
} from 'react-native';
import { useTheme } from '../theme';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  style?: ViewStyle;
}

export default function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon,
  style,
}: ButtonProps) {
  const { theme } = useTheme();

  const bgColor = {
    primary: theme.colors.primary,
    secondary: theme.colors.surfaceLight,
    ghost: 'transparent',
    danger: theme.colors.recording,
  }[variant];

  const textColor = {
    primary: theme.colors.onPrimary,
    secondary: theme.colors.onSurface,
    ghost: theme.colors.primary,
    danger: theme.colors.onPrimary,
  }[variant];

  const borderColor = {
    primary: 'transparent',
    secondary: theme.colors.borderLight,
    ghost: theme.colors.primary,
    danger: 'transparent',
  }[variant];

  const paddingVertical = {
    sm: theme.spacing.sm,
    md: theme.spacing.md,
    lg: theme.spacing.lg,
  }[size];

  const fontSize = {
    sm: theme.typography.caption.fontSize,
    md: theme.typography.body.fontSize,
    lg: theme.typography.h4.fontSize,
  }[size];

  const borderRadius = {
    sm: theme.radii.md,
    md: theme.radii.lg,
    lg: theme.radii.xl,
  }[size];

  return (
    <TouchableOpacity
      style={[
        styles.button,
        {
          backgroundColor: disabled ? theme.colors.border : bgColor,
          borderColor,
          borderWidth: variant === 'ghost' ? 1.5 : 0,
          paddingVertical,
          paddingHorizontal: paddingVertical * 2,
          borderRadius,
        },
        style,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator color={textColor} size="small" />
      ) : (
        <>
          {icon}
          <Text
            style={[
              styles.text,
              {
                color: disabled ? theme.colors.textTertiary : textColor,
                fontSize,
                fontWeight: theme.typography.bodyBold.fontWeight,
              } as TextStyle,
            ]}
          >
            {title}
          </Text>
        </>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  text: {
    textAlign: 'center',
  },
});
