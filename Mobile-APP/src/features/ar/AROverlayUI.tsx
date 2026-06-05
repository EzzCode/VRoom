import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useTheme } from '../../shared/theme';
import { IconButton, TrackingIndicator } from '../../shared/components';

type InteractionMode = 'place' | 'move-floor' | 'move-lift' | 'rotate-horiz' | 'rotate-vert' | 'rotate-roll' | 'scale';
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
  currentScale: number;
  onScaleChange: (scale: number) => void;
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
  currentScale,
  onScaleChange,
}: AROverlayUIProps) {
  const { theme } = useTheme();

  const modes: { key: InteractionMode; label: string }[] = [
    { key: 'place', label: 'Place' },
    { key: 'move-floor', label: 'Move' },
    { key: 'rotate-horiz', label: 'Rotate' },
    { key: 'scale', label: 'Scale' },
  ];

  const moveSubModes: { key: InteractionMode; label: string }[] = [
    { key: 'move-floor', label: 'Floor' },
    { key: 'move-lift', label: 'Lift' },
  ];

  const rotateSubModes: { key: InteractionMode; label: string }[] = [
    { key: 'rotate-horiz', label: 'Spin' },
    { key: 'rotate-vert', label: 'Tilt' },
    { key: 'rotate-roll', label: 'Roll' },
  ];

  const isMoveActive = interactionMode === 'move-floor' || interactionMode === 'move-lift';
  const isRotateActive =
    interactionMode === 'rotate-horiz' ||
    interactionMode === 'rotate-vert' ||
    interactionMode === 'rotate-roll';
  // Highlight the correct main button
  const effectiveModeKey = isMoveActive
    ? 'move-floor'
    : isRotateActive
      ? 'rotate-horiz'
      : interactionMode;

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
            Please move your phone to initialize AR
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
            Looking for a flat surface to place your object...
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
                ? 'Tap the target to place your object'
                : 'Move closer to a flat surface'}
            </Text>
          </View>
        </View>
      )}

      {/* Mode selector */}
      {isMeshPlaced && (
        <View style={styles.modeBar} pointerEvents="box-none">
          {/* Move sub-bar */}
          {isMoveActive && (
            <View
              style={[
                styles.modeBarInner,
                {
                  backgroundColor: theme.colors.overlay,
                  borderRadius: theme.radii.xl,
                  paddingVertical: theme.spacing.sm,
                  paddingHorizontal: theme.spacing.sm,
                  marginBottom: theme.spacing.sm,
                },
              ]}
              pointerEvents="box-none"
            >
              {moveSubModes.map((sub) => (
                <TouchableOpacity
                  key={sub.key}
                  style={[
                    styles.modeButton,
                    {
                      backgroundColor:
                        interactionMode === sub.key ? theme.colors.primary : 'transparent',
                      borderRadius: theme.radii.lg,
                      paddingHorizontal: theme.spacing.lg,
                      paddingVertical: theme.spacing.sm,
                    },
                  ]}
                  onPress={() => setInteractionMode(sub.key)}
                >
                  <Text
                    style={{
                      color: interactionMode === sub.key ? '#FFFFFF' : theme.colors.textSecondary,
                      fontSize: theme.typography.caption.fontSize,
                      fontWeight: interactionMode === sub.key ? '700' : '400',
                    }}
                  >
                    {sub.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* Rotate sub-bar */}
          {isRotateActive && (
            <View
              style={[
                styles.modeBarInner,
                {
                  backgroundColor: theme.colors.overlay,
                  borderRadius: theme.radii.xl,
                  paddingVertical: theme.spacing.sm,
                  paddingHorizontal: theme.spacing.sm,
                  marginBottom: theme.spacing.sm,
                },
              ]}
              pointerEvents="box-none"
            >
              {rotateSubModes.map((sub) => (
                <TouchableOpacity
                  key={sub.key}
                  style={[
                    styles.modeButton,
                    {
                      backgroundColor:
                        interactionMode === sub.key ? theme.colors.primary : 'transparent',
                      borderRadius: theme.radii.lg,
                      paddingHorizontal: theme.spacing.lg,
                      paddingVertical: theme.spacing.sm,
                    },
                  ]}
                  onPress={() => setInteractionMode(sub.key)}
                >
                  <Text
                    style={{
                      color: interactionMode === sub.key ? '#FFFFFF' : theme.colors.textSecondary,
                      fontSize: theme.typography.caption.fontSize,
                      fontWeight: interactionMode === sub.key ? '700' : '400',
                    }}
                  >
                    {sub.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* Scale +/- buttons */}
          {interactionMode === 'scale' && (
            <View style={[styles.scalePill, { backgroundColor: theme.colors.overlay, borderRadius: theme.radii.xl }]}>
              <TouchableOpacity
                style={[styles.scaleBtn, { backgroundColor: theme.colors.primary }]}
                onPress={() => onScaleChange(Math.max(0.05, currentScale / 1.1))}
              >
                <Text style={{ color: '#FFFFFF', fontSize: 22, lineHeight: 26, fontWeight: '300' }}>−</Text>
              </TouchableOpacity>
              <Text style={{ color: '#FFFFFF', fontSize: 13, fontWeight: '600', minWidth: 52, textAlign: 'center' }}>
                {currentScale.toFixed(2)}×
              </Text>
              <TouchableOpacity
                style={[styles.scaleBtn, { backgroundColor: theme.colors.primary }]}
                onPress={() => onScaleChange(Math.min(3.0, currentScale * 1.1))}
              >
                <Text style={{ color: '#FFFFFF', fontSize: 22, lineHeight: 26, fontWeight: '300' }}>+</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Main mode bar */}
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
                      effectiveModeKey === mode.key ? theme.colors.primary : 'transparent',
                    borderRadius: theme.radii.lg,
                    paddingHorizontal: theme.spacing.lg,
                    paddingVertical: theme.spacing.sm,
                  },
                ]}
                onPress={() => setInteractionMode(mode.key)}
              >
                <Text
                  style={{
                    color: effectiveModeKey === mode.key ? '#FFFFFF' : theme.colors.textSecondary,
                    fontSize: theme.typography.caption.fontSize,
                    fontWeight: effectiveModeKey === mode.key ? '700' : '400',
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
  topBar: {
    width: '100%',
  },
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  topCenter: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  banner: {
    alignSelf: 'stretch',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingCard: {
    alignItems: 'center',
  },
  tooltip: {
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
  scalePill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 8,
    marginBottom: 10,
    gap: 4,
  },
  scaleBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
