import React from 'react';
import { TouchableOpacity, StyleSheet } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useTheme } from '../theme';

interface IconButtonProps {
  icon: keyof typeof Ionicons.glyphMap | string;
  onPress: () => void;
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  disabled?: boolean;
  style?: any;
}

export default function IconButton({
  icon,
  onPress,
  size = 'md',
  color,
  disabled = false,
  style,
}: IconButtonProps) {
  const { theme } = useTheme();

  const iconSize = { sm: 20, md: 24, lg: 32 }[size];
  const hitSlop = { sm: 8, md: 12, lg: 16 }[size];
  const iconColor = color ?? theme.colors.textPrimary;

  return (
    <TouchableOpacity
      style={[styles.button, style]}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.6}
      hitSlop={{ top: hitSlop, bottom: hitSlop, left: hitSlop, right: hitSlop }}
    >
      <Ionicons
        name={icon as any}
        size={iconSize}
        color={disabled ? theme.colors.textTertiary : iconColor}
      />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});
