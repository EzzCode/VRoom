import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useTheme } from '../theme';

interface TrackingIndicatorProps {
  state: 'normal' | 'limited' | 'unavailable';
  size?: number;
}

export default function TrackingIndicator({ state, size = 10 }: TrackingIndicatorProps) {
  const { theme } = useTheme();

  const color = {
    normal: theme.colors.trackingNormal,
    limited: theme.colors.trackingLimited,
    unavailable: theme.colors.trackingUnavailable,
  }[state];

  return (
    <View
      style={[
        styles.container,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: color,
        },
      ]}
    />
  );
}

const styles = StyleSheet.create({
  container: {},
});
