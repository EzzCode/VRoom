import React, { useEffect } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import { useTheme } from '../theme';

interface ProgressBarProps {
  progress: number;
  animated?: boolean;
  color?: string;
  height?: number;
  trackColor?: string;
}

export default function ProgressBar({
  progress,
  animated = true,
  color,
  height = 4,
  trackColor,
}: ProgressBarProps) {
  const { theme } = useTheme();
  const animatedWidth = React.useRef(new Animated.Value(0)).current;
  const clampedProgress = Math.max(0, Math.min(1, progress));

  useEffect(() => {
    if (animated) {
      Animated.spring(animatedWidth, {
        toValue: clampedProgress,
        useNativeDriver: false,
        speed: 2,
        bounciness: 0,
      }).start();
    }
  }, [clampedProgress, animated, animatedWidth]);

  return (
    <View
      style={[
        styles.track,
        {
          backgroundColor: trackColor ?? theme.colors.border,
          height,
          borderRadius: height / 2,
        },
      ]}
    >
      <Animated.View
        style={[
          styles.fill,
          {
            backgroundColor: color ?? theme.colors.primary,
            height,
            borderRadius: height / 2,
            width: animated
              ? animatedWidth.interpolate({
                  inputRange: [0, 1],
                  outputRange: ['0%', '100%'],
                })
              : `${clampedProgress * 100}%`,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  track: {
    overflow: 'hidden',
  },
  fill: {},
});
