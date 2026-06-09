import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Modal,
  FlatList,
  SafeAreaView,
  Image,
} from 'react-native';
import { useTheme } from '../../shared/theme';
import { IconButton, TrackingIndicator } from '../../shared/components';
import { MeshInfo } from '../../shared/core/types';
import { RoomLayout, loadLayouts, deleteLayout } from '../../services/mesh/layoutStorage';
import { InteractionMode, TrackingState } from './arTypes';

interface AROverlayUIProps {
  onBack: () => void;
  onScreenshot: () => void;
  onReset: () => void;
  /** Name of the currently active (selected) mesh */
  activeMeshName: string;
  interactionMode: InteractionMode;
  setInteractionMode: (mode: InteractionMode) => void;
  trackingState: TrackingState;
  /** At least one mesh has been placed in the scene */
  anyMeshPlaced: boolean;
  /** There is currently a mesh waiting to be placed */
  hasUnplacedMesh: boolean;
  isMeshLoading: boolean;
  reticleVisible: boolean;
  currentScale: number;
  onScaleChange: (scale: number) => void;
  /** All meshes available in the library (PLY filtered out by parent) */
  availableMeshes: MeshInfo[];
  onAddMesh: (mesh: MeshInfo) => void;
  onSaveLayout?: (name: string) => void;
  onLoadLayout?: (layout: RoomLayout) => void;
  aligningLayout?: RoomLayout;
  onConfirmAlignment?: () => void;
  onCancelAlignment?: () => void;
}

export default function AROverlayUI({
  onBack,
  onScreenshot,
  onReset,
  activeMeshName,
  interactionMode,
  setInteractionMode,
  trackingState,
  anyMeshPlaced,
  hasUnplacedMesh,
  isMeshLoading,
  reticleVisible,
  currentScale,
  onScaleChange,
  availableMeshes,
  onAddMesh,
  onSaveLayout,
  onLoadLayout,
  aligningLayout,
  onConfirmAlignment,
  onCancelAlignment,
}: AROverlayUIProps) {
  const { theme } = useTheme();
  const [showMeshPicker, setShowMeshPicker] = useState(false);
  
  // For the ghost image alignment
  const [ghostOpacity, setGhostOpacity] = useState(0.4);
  
  const [showLayoutsModal, setShowLayoutsModal] = useState(false);
  const [savedLayouts, setSavedLayouts] = useState<RoomLayout[]>([]);
  const [isLayoutsLoading, setIsLayoutsLoading] = useState(false);

  const fetchLayouts = async () => {
    setIsLayoutsLoading(true);
    const layouts = await loadLayouts();
    setSavedLayouts(layouts);
    setIsLayoutsLoading(false);
  };

  const handleOpenLayouts = () => {
    setShowLayoutsModal(true);
    fetchLayouts();
  };

  const handleDeleteLayout = async (id: string) => {
    await deleteLayout(id);
    fetchLayouts();
  };

  // Main mode bar — Place is auto-set, not a button
  const modes: { key: InteractionMode; label: string }[] = [
    { key: 'select', label: 'Select' },
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

  // Which main bar button to highlight
  const effectiveModeKey: InteractionMode = isMoveActive
    ? 'move-floor'
    : isRotateActive
      ? 'rotate-horiz'
      : interactionMode;

  // AR-compatible meshes for the picker (PLY cannot be displayed)
  const arCompatibleMeshes = availableMeshes.filter((m) => m.format !== 'PLY');

  if (aligningLayout) {
    return (
      <View style={styles.container} pointerEvents="box-none">
        {/* Ghost Image at adjustable opacity */}
        {aligningLayout.screenshotUri && (
          <Image
            source={{ uri: aligningLayout.screenshotUri }}
            style={[StyleSheet.absoluteFill, { opacity: ghostOpacity }]}
            resizeMode="cover"
          />
        )}

        {/* Top Instruction Banner */}
        <SafeAreaView style={{ flex: 1, justifyContent: 'space-between' }} pointerEvents="box-none">
          <View style={{ alignItems: 'center', marginTop: 60 }} pointerEvents="box-none">
            <View style={{ backgroundColor: 'rgba(0,0,0,0.8)', paddingHorizontal: 24, paddingVertical: 16, borderRadius: theme.radii.lg, alignItems: 'center' }}>
              <Text style={{ color: '#FFFFFF', fontSize: 18, fontWeight: '700', marginBottom: 4 }}>Align Your Camera</Text>
              <Text style={{ color: '#EEEEEE', fontSize: 14, textAlign: 'center', marginBottom: 12 }}>
                Move your phone to match the reference image.
              </Text>
              
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 16 }}>
                <TouchableOpacity 
                  onPress={() => setGhostOpacity(Math.max(0.1, ghostOpacity - 0.1))}
                  style={{ backgroundColor: 'rgba(255,255,255,0.2)', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' }}
                >
                  <Text style={{ color: '#FFF', fontSize: 20, fontWeight: 'bold' }}>-</Text>
                </TouchableOpacity>
                <Text style={{ color: '#FFF', fontSize: 16, fontWeight: '600', width: 80, textAlign: 'center' }}>
                  Opacity {Math.round(ghostOpacity * 100)}%
                </Text>
                <TouchableOpacity 
                  onPress={() => setGhostOpacity(Math.min(1.0, ghostOpacity + 0.1))}
                  style={{ backgroundColor: 'rgba(255,255,255,0.2)', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' }}
                >
                  <Text style={{ color: '#FFF', fontSize: 20, fontWeight: 'bold' }}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          {/* Bottom Action Bar */}
          <View style={{ paddingHorizontal: theme.spacing.xl, paddingBottom: 50, gap: 16 }} pointerEvents="box-none">

            <TouchableOpacity
              style={{ backgroundColor: theme.colors.primary, paddingVertical: 16, borderRadius: theme.radii.lg, alignItems: 'center' }}
              onPress={onConfirmAlignment}
            >
              <Text style={{ color: '#FFFFFF', fontSize: 16, fontWeight: '700' }}>Confirm Alignment</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={{ backgroundColor: 'rgba(255,255,255,0.2)', paddingVertical: 16, borderRadius: theme.radii.lg, alignItems: 'center' }}
              onPress={onCancelAlignment}
            >
              <Text style={{ color: '#FFFFFF', fontSize: 16, fontWeight: '600' }}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.container} pointerEvents="box-none">
      {/* ── Top Bar ─────────────────────────────────────────────────────────── */}
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
              {activeMeshName}
            </Text>
          </View>
          <TrackingIndicator state={trackingState} size={12} />
        </View>
      </View>

      {/* ── Tracking banners ────────────────────────────────────────────────── */}
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

      {/* ── Loading overlay ──────────────────────────────────────────────────── */}
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

      {/* ── Placement guidance ───────────────────────────────────────────────── */}
      {hasUnplacedMesh && trackingState === 'normal' && (
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
                ? `Tap to place "${activeMeshName}"`
                : 'Move closer to a flat surface'}
            </Text>
          </View>
        </View>
      )}

      {/* ── Mode selector (shown once at least one mesh is placed) ──────────── */}
      {anyMeshPlaced && (
        <View style={styles.modeBar} pointerEvents="box-none">
          {/* Direct-manipulation hint */}
          <View
            style={[
              styles.gestureHint,
              { backgroundColor: theme.colors.overlay, borderRadius: theme.radii.xl },
            ]}
            pointerEvents="none"
          >
            <Text
              style={{
                color: theme.colors.textSecondary,
                fontSize: theme.typography.caption.fontSize,
                fontWeight: '600',
              }}
            >
              Pinch to scale · twist to rotate
            </Text>
          </View>
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
                      color:
                        interactionMode === sub.key ? '#FFFFFF' : theme.colors.textSecondary,
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
                      color:
                        interactionMode === sub.key ? '#FFFFFF' : theme.colors.textSecondary,
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
            <View
              style={[
                styles.scalePill,
                { backgroundColor: theme.colors.overlay, borderRadius: theme.radii.xl },
              ]}
            >
              <TouchableOpacity
                style={[styles.scaleBtn, { backgroundColor: theme.colors.primary }]}
                onPress={() => onScaleChange(Math.max(0.05, currentScale / 1.1))}
              >
                <Text style={{ color: '#FFFFFF', fontSize: 22, lineHeight: 26, fontWeight: '300' }}>
                  −
                </Text>
              </TouchableOpacity>
              <Text
                style={{
                  color: '#FFFFFF',
                  fontSize: 13,
                  fontWeight: '600',
                  minWidth: 52,
                  textAlign: 'center',
                }}
              >
                {currentScale.toFixed(2)}×
              </Text>
              <TouchableOpacity
                style={[styles.scaleBtn, { backgroundColor: theme.colors.primary }]}
                onPress={() => onScaleChange(Math.min(3.0, currentScale * 1.1))}
              >
                <Text style={{ color: '#FFFFFF', fontSize: 22, lineHeight: 26, fontWeight: '300' }}>
                  +
                </Text>
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
                    color:
                      effectiveModeKey === mode.key ? '#FFFFFF' : theme.colors.textSecondary,
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

        <View style={{ flexDirection: 'row', gap: 12 }}>
          {/* Add Object button */}
          <TouchableOpacity
            style={[
              styles.addButton,
              { backgroundColor: theme.colors.primary, borderRadius: theme.radii.xl },
            ]}
            onPress={() => setShowMeshPicker(true)}
          >
            <Text style={{ color: '#FFFFFF', fontSize: 22, lineHeight: 26, fontWeight: '300' }}>
              +
            </Text>
            <Text
              style={{
                color: '#FFFFFF',
                fontSize: theme.typography.caption.fontSize,
                fontWeight: '600',
                marginLeft: 4,
              }}
            >
              Add Object
            </Text>
          </TouchableOpacity>

          {/* Layouts button */}
          <TouchableOpacity
            style={[
              styles.addButton,
              { backgroundColor: theme.colors.surface, borderRadius: theme.radii.xl },
            ]}
            onPress={handleOpenLayouts}
          >
            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.caption.fontSize,
                fontWeight: '600',
              }}
            >
              Layouts
            </Text>
          </TouchableOpacity>
        </View>

        {anyMeshPlaced && (
          <IconButton icon="refresh-outline" onPress={onReset} color="#FFFFFF" size="lg" />
        )}
      </View>

      {/* ── Mesh Picker Modal ────────────────────────────────────────────────── */}
      <Modal
        visible={showMeshPicker}
        transparent
        animationType="slide"
        onRequestClose={() => setShowMeshPicker(false)}
      >
        <TouchableOpacity
          style={styles.modalBackdrop}
          activeOpacity={1}
          onPress={() => setShowMeshPicker(false)}
        />
        <SafeAreaView style={styles.modalSheet}>
          <View
            style={[
              styles.modalContent,
              { backgroundColor: theme.colors.surface },
            ]}
          >
            {/* Handle bar */}
            <View
              style={[styles.modalHandle, { backgroundColor: theme.colors.textTertiary }]}
            />

            <Text
              style={{
                color: theme.colors.textPrimary,
                fontSize: theme.typography.h4.fontSize,
                fontWeight: '700',
                marginBottom: theme.spacing.md,
                paddingHorizontal: theme.spacing.lg,
              }}
            >
              Add Object
            </Text>

            {arCompatibleMeshes.length === 0 ? (
              <View style={styles.emptyState}>
                <Text
                  style={{
                    color: theme.colors.textTertiary,
                    fontSize: theme.typography.body.fontSize,
                    textAlign: 'center',
                    paddingHorizontal: theme.spacing.xl,
                  }}
                >
                  No objects in your library.{'\n'}Import a .glb or .obj file from the Mesh
                  Gallery.
                </Text>
              </View>
            ) : (
              <FlatList
                data={arCompatibleMeshes}
                keyExtractor={(item) => item.id}
                style={{ maxHeight: 320 }}
                contentContainerStyle={{ paddingHorizontal: theme.spacing.lg }}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[
                      styles.meshPickerItem,
                      {
                        borderBottomColor: theme.colors.border,
                      },
                    ]}
                    onPress={() => {
                      setShowMeshPicker(false);
                      onAddMesh(item);
                    }}
                  >
                    <View style={{ flex: 1 }}>
                      <Text
                        style={{
                          color: theme.colors.textPrimary,
                          fontSize: theme.typography.body.fontSize,
                          fontWeight: '600',
                        }}
                        numberOfLines={1}
                      >
                        {item.name}
                      </Text>
                      <Text
                        style={{
                          color: theme.colors.textSecondary,
                          fontSize: theme.typography.caption.fontSize,
                          marginTop: 2,
                        }}
                      >
                        {item.format}
                      </Text>
                    </View>
                    <View
                      style={[
                        styles.formatBadge,
                        { backgroundColor: theme.colors.primary + '22' },
                      ]}
                    >
                      <Text
                        style={{
                          color: theme.colors.primary,
                          fontSize: 11,
                          fontWeight: '700',
                        }}
                      >
                        {item.format}
                      </Text>
                    </View>
                  </TouchableOpacity>
                )}
              />
            )}

            <TouchableOpacity
              style={[
                styles.cancelButton,
                {
                  marginHorizontal: theme.spacing.lg,
                  marginTop: theme.spacing.md,
                  marginBottom: theme.spacing.sm,
                  borderRadius: theme.radii.lg,
                  backgroundColor: theme.colors.errorBackground ?? 'rgba(255,80,80,0.12)',
                },
              ]}
              onPress={() => setShowMeshPicker(false)}
            >
              <Text
                style={{
                  color: theme.colors.error,
                  fontSize: theme.typography.body.fontSize,
                  fontWeight: '600',
                  textAlign: 'center',
                }}
              >
                Cancel
              </Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>

      {/* ── Layouts Modal ─────────────────────────────────────────────────────── */}
      <Modal
        visible={showLayoutsModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowLayoutsModal(false)}
      >
        <TouchableOpacity
          style={styles.modalBackdrop}
          activeOpacity={1}
          onPress={() => setShowLayoutsModal(false)}
        />
        <SafeAreaView style={styles.modalSheet}>
          <View style={[styles.modalContent, { backgroundColor: theme.colors.surface }]}>
            <View style={[styles.modalHandle, { backgroundColor: theme.colors.textTertiary }]} />

            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: theme.spacing.lg, marginBottom: theme.spacing.md }}>
              <Text
                style={{
                  color: theme.colors.textPrimary,
                  fontSize: theme.typography.h4.fontSize,
                  fontWeight: '700',
                }}
              >
                Room Layouts
              </Text>
              <TouchableOpacity
                style={{
                  backgroundColor: theme.colors.primary,
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: theme.radii.md,
                }}
                onPress={() => {
                  if (onSaveLayout) {
                    onSaveLayout('');
                    // give it a tiny delay to save before fetching
                    setTimeout(fetchLayouts, 500);
                  }
                }}
              >
                <Text style={{ color: '#FFFFFF', fontWeight: '600' }}>Save Current</Text>
              </TouchableOpacity>
            </View>

            {isLayoutsLoading ? (
              <ActivityIndicator style={{ paddingVertical: 40 }} color={theme.colors.primary} />
            ) : savedLayouts.length === 0 ? (
              <View style={styles.emptyState}>
                <Text style={{ color: theme.colors.textTertiary, textAlign: 'center' }}>
                  No saved layouts yet.{'\n'}Place some objects and save your layout!
                </Text>
              </View>
            ) : (
              <FlatList
                data={savedLayouts}
                keyExtractor={(item) => item.id}
                style={{ maxHeight: 320 }}
                contentContainerStyle={{ paddingHorizontal: theme.spacing.lg }}
                renderItem={({ item }) => (
                  <View style={[styles.meshPickerItem, { borderBottomColor: theme.colors.border }]}>
                    <View style={{ flex: 1 }}>
                      <Text style={{ color: theme.colors.textPrimary, fontWeight: '600' }} numberOfLines={1}>
                        {item.name}
                      </Text>
                      <Text style={{ color: theme.colors.textSecondary, fontSize: 12, marginTop: 2 }}>
                        {item.meshes.length} object{item.meshes.length !== 1 ? 's' : ''} • {new Date(item.createdAt).toLocaleDateString()}
                      </Text>
                    </View>
                    <View style={{ flexDirection: 'row', gap: 8 }}>
                      <TouchableOpacity
                        style={[styles.formatBadge, { backgroundColor: theme.colors.errorBackground }]}
                        onPress={() => handleDeleteLayout(item.id)}
                      >
                        <Text style={{ color: theme.colors.error, fontSize: 12, fontWeight: '600' }}>Delete</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={[styles.formatBadge, { backgroundColor: theme.colors.primary + '22' }]}
                        onPress={() => {
                          setShowLayoutsModal(false);
                          if (onLoadLayout) onLoadLayout(item);
                        }}
                      >
                        <Text style={{ color: theme.colors.primary, fontSize: 12, fontWeight: '600' }}>Load</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
              />
            )}

            <TouchableOpacity
              style={[
                styles.cancelButton,
                {
                  marginHorizontal: theme.spacing.lg,
                  marginTop: theme.spacing.md,
                  marginBottom: theme.spacing.sm,
                  borderRadius: theme.radii.lg,
                  backgroundColor: theme.colors.errorBackground ?? 'rgba(255,80,80,0.12)',
                },
              ]}
              onPress={() => setShowLayoutsModal(false)}
            >
              <Text style={{ color: theme.colors.error, fontWeight: '600', textAlign: 'center' }}>
                Close
              </Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>
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
  gestureHint: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    marginBottom: 8,
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
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
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
  // Modal
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalSheet: {
    backgroundColor: 'transparent',
  },
  modalContent: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingTop: 12,
    paddingBottom: 8,
  },
  modalHandle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: 16,
  },
  emptyState: {
    paddingVertical: 40,
    alignItems: 'center',
  },
  meshPickerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  formatBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginLeft: 8,
  },
  cancelButton: {
    paddingVertical: 14,
  },
});
