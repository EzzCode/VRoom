# Coverage Feature

Voxel-based tracking of which parts of the captured volume have actually been observed by the camera.

## Modules

| File | Purpose |
|---|---|
| `CoverageTracker.ts` | Pure module: maintains the voxel map, accepts `CameraPose`s, reports `coveragePercent`. Has both mutating `observe(pose)` and non-mutating `peek(pose)`. |
| `__tests__/CoverageTracker.test.ts` | Unit tests for the tracker (Jest, no React deps). |

The matching gate lives at [features/capture/gates/CoverageGate.ts](../capture/gates/CoverageGate.ts) — it calls `tracker.peek(pose)` to gate keyframe acceptance. The tracker instance is owned by [providers/SessionProvider.tsx](../../providers/SessionProvider.tsx) so the gate, the HUD, and (later) the AR overlay share one source of truth.

## Voxel state machine

```
unseen   → not in map
partial  → in map, observationCount < CAPTURE_CONFIG.coverage.minObservations
covered  → in map, observationCount >= CAPTURE_CONFIG.coverage.minObservations
```

`coveragePercent` = covered / touched. It represents the quality of what we have, not progress against an unknown total volume.

## Known gap: camera pose source

`SessionProvider.setCurrentPose` is **not yet called by anyone**. `CaptureScreen` only mounts `<Camera>` from `react-native-vision-camera`, not a `ViroARScene`. Until a pose source lands, both `AngleGate` and `CoverageGate` see `pose === null` and pass by default — coverage % will stay at 0.

Two ways to close the gap:

1. Mount a hidden `ViroARSceneNavigator` alongside the camera and feed pose updates into `setCurrentPose`. Tricky: ARKit/ARCore and VisionCamera may fight for the camera.
2. Use `expo-sensors` (`DeviceMotion`) as a coarse pose proxy. Loses translation, only gives rotation — adequate for orientation-based coverage but not position-aware voxel marking.

Option 1 is the right one for full coverage tracking. To be tackled as a follow-up sub-task.

## Next steps (Build 3 remainder)

- `VoxelOverlay.tsx`: render `tracker.getVoxels()` as semi-transparent `ViroBox` cubes (yellow for partial, green for covered) in the AR scene.
- Guidance hints: when several seconds pass with no `newlyCovered`, surface a "Move to an uncovered area" prompt using the dominant axis of unseen voxels relative to current camera forward.
