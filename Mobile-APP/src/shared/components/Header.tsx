import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../theme';
import IconButton from './IconButton';

interface HeaderProps {
  title?: string;
  onBack?: () => void;
  rightAction?: React.ReactNode;
  transparent?: boolean;
}

export default function Header({ title, onBack, rightAction, transparent = false }: HeaderProps) {
  const { theme } = useTheme();

  return (
    <View
      style={[
        styles.container,
        {
          paddingTop: theme.spacing.xl,
          paddingBottom: theme.spacing.md,
          paddingHorizontal: theme.spacing.lg,
          backgroundColor: transparent ? 'transparent' : theme.colors.background,
        },
      ]}
    >
      <View style={styles.left}>
        {onBack && <IconButton icon="arrow-back" onPress={onBack} size="sm" />}
      </View>

      <View style={styles.center}>
        {title && (
          <Text
            style={[
              styles.title,
              {
                color: theme.colors.textPrimary,
                fontSize: theme.typography.h3.fontSize,
                fontWeight: theme.typography.h3.fontWeight,
              },
            ]}
            numberOfLines={1}
          >
            {title}
          </Text>
        )}
      </View>

      <View style={styles.right}>{rightAction}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  left: {
    width: 40,
    alignItems: 'flex-start',
  },
  center: {
    flex: 1,
    alignItems: 'center',
  },
  right: {
    width: 40,
    alignItems: 'flex-end',
  },
  title: {},
});
