import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useTheme } from '../../shared/theme';
import { IconButton, TrackingIndicator } from '../../shared/components';

type InteractionMode = 'place' | 'move' | 'rotate' | 'scale';
type TrackingState = 'unavailable' | 'limited' | 'normal';

interface AROverlayUIProps {
  onBack: () => void;
  onScreenshot: () => void;
  onReset: () => void;
  meshName: string;
  interactionMode: InteractionMode;
  setInteractionMode: (mode: InteractionMode) => void;
  trackingState: TrackingState;
  isMeshPlaced: boolean;
  isMeshLoading: boolean;
  reticleVisible: boolean;
}

export default function AROverlayUI({
  onBack,
  onScreenshot,
  onReset,
  meshName,
  interactionMode,
  setInteractionMode,
  trackingState,
  isMeshPlaced,
  isMeshLoading,
  reticleVisible,
}: AROverlayUIProps) {
  const { theme } = useTheme();

  const modes: { key: InteractionMode; label: string }[] = [
    { key: 'place', label: 'Place' },
    { key: 'move', label: 'Move' },
    { key: 'rotate', label: 'Rotate' },
    { key: 'scale', label: 'Scale' },
  ];

  return (
    <View style={styles.container} pointerEvents="box-none">
      {/* Top Bar */}
      <View
        style={[
          styles.topBar,
          {
            paddingTop: 50,
            paddingHorizontal: theme.spacing.lg,
            paddingBottom: theme.spacing.md,
          },
        ]}
        pointerEvents="box-none"
      >
        <View style={styles.topRow} pointerEvents="box-none">
          <IconButton icon="arrow-back" onPress={onBack} color="#FFFFFF" />
          <View style={styles.topCenter}>
            <Text
              style={{
                color: '#FFFFFF',
                fontSize: theme.typography.body.fontSize,
                fontWeight: '600',
              }}
              numberOfLines={1}
            >
              {meshName}
            </Text>
          </View>
          <TrackingIndicator state={trackingState} size={12} />
        </View>
      </View>

      {/* Tracking guidance */}
      {trackingState === 'unavailable' && (
        <View
          style={[
            styles.banner,
            {
              backgroundColor: theme.colors.errorBackground,
              marginHorizontal: theme.spacing.lg,
              borderRadius: theme.radii.md,
              paddingVertical: theme.spacing.md,
              paddingHorizontal: theme.spacing.xl,
              borderLeftWidth: 3,
              borderLeftColor: theme.colors.error,
            },
          ]}
          pointerEvents="none"
        >
          <Text
            style={{
              color: theme.colors.error,
              fontSize: theme.typography.body.fontSize,
              fontWeight: '600',
              textAlign: 'center',
            }}
          >
            Move your phone around to start AR tracking
          </Text>
        </View>
      )}

      {trackingState === 'limited' && (
        <View
          style={[
            styles.banner,
            {
              backgroundColor: theme.colors.warningBackground,
              marginHorizontal: theme.spacing.lg,
              borderRadius: theme.radii.md,
              paddingVertical: theme.spacing.md,
              paddingHorizontal: theme.spacing.xl,
              borderLeftWidth: 3,
              borderLeftColor: theme.colors.warning,
            },
          ]}
          pointerEvents="none"
        >
          <Text
            style={{
              color: theme.colors.warning,
              fontSize: theme.typography.caption.fontSize,
              fontWeight: '600',
              textAlign: 'center',
            }}
          >
            Point your camera at a textured flat surface (floor, table, desk)
          </Text>
        </View>
      )}

      {/* Loading overlay */}
      {isMeshLoading && (
        <View style={styles.loadingOverlay} pointerEvents="none">
          <View
            style={[
              styles.loadingCard,
              {
                backgroundColor: theme.colors.overlay,
                borderRadius: theme.radii.lg,
                padding: theme.spacing.xl,
              },
            ]}
          >
            <ActivityIndicator color={theme.colors.primary} size="large" />
            <Text
              style={{
                color: '#FFFFFF',
                fontSize: theme.typography.body.fontSize,
                marginTop: theme.spacing.md,
              }}
            >
              Loading mesh...
            </Text>
          </View>
        </View>
      )}

      {/* Placement guidance */}
      {!isMeshPlaced && trackingState === 'normal' && (
        <View style={styles.tooltip} pointerEvents="none">
          <View
            style={[
              styles.tooltipCard,
              {
                backgroundColor: theme.colors.overlay,
                borderRadius: theme.radii.md,
                paddingVertical: theme.spacing.md,
                paddingHorizontal: theme.spacing.xl,
              },
            ]}
          >
            <Text
              style={{
                color: '#FFFFFF',
                fontSize: theme.typography.body.fontSize,
                textAlign: 'center',
              }}
            >
              {reticleVisible
                ? 'Tap to place your mesh here'
                : 'Point your camera at a flat surface'}
            </Text>
          </View>
        </View>
      )}

      {/* Mode selector */}
      {isMeshPlaced && (
        <View style={styles.modeBar} pointerEvents="box-none">
          <View
            style={[
              styles.modeBarInner,
              {
                backgroundColor: theme.colors.overlay,
                borderRadius: theme.radii.xl,
                paddingVertical: theme.spacing.sm,
                paddingHorizontal: theme.spacing.sm,
              },
            ]}
            pointerEvents="box-none"
          >
            {modes.map((mode) => (
              <TouchableOpacity
                key={mode.key}
                style={[
                  styles.modeButton,
                  {
                    backgroundColor:
                      interactionMode === mode.key ? theme.colors.primary : 'transparent',
                    borderRadius: theme.radii.lg,
                    paddingHorizontal: theme.spacing.lg,
                    paddingVertical: theme.spacing.sm,
                  },
                ]}
                onPress={() => setInteractionMode(mode.key)}
              >
                <Text
                  style={{
                    color: interactionMode === mode.key ? '#FFFFFF' : theme.colors.textSecondary,
                    fontSize: theme.typography.caption.fontSize,
                    fontWeight: interactionMode === mode.key ? '700' : '400',
                  }}
                >
                  {mode.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* Bottom bar */}
      <View
        style={[
          styles.bottomBar,
          {
            paddingBottom: 40,
            paddingHorizontal: theme.spacing.xl,
          },
        ]}
        pointerEvents="box-none"
      >
        <IconButton icon="camera-outline" onPress={onScreenshot} color="#FFFFFF" size="lg" />
        {isMeshPlaced && (
          <IconButton icon="refresh-outline" onPress={onReset} color="#FFFFFF" size="lg" />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
  },
  topBar: {},
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  topCenter: {
    flex: 1,
    alignItems: 'center',
  },
  banner: {},
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingCard: {
    alignItems: 'center',
  },
  tooltip: {
    position: 'absolute',
    top: '40%',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  tooltipCard: {},
  modeBar: {
    position: 'absolute',
    bottom: 120,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  modeBarInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  modeButton: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
});
