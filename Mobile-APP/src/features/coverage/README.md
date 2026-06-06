# Coverage Feature

Voxel-based tracking of which parts of a scene have actually been observed by the camera.

## Modules

| File | Purpose |
|---|---|
| `CoverageTracker.ts` | Pure module (no React/Viro/IO): maintains the voxel map and reports `coveragePercent`. `observe(pose)` marks the camera frustum volume; `observePoint(worldPoint)` marks the single voxel at a surface point; `peek(pose)` is the non-mutating prediction used by the gate; `getVoxels()` snapshots voxels for rendering. |
| `VoxelOverlay.tsx` | Renders `getVoxels()` as translucent `ViroBox` cubes (yellow = partial, green = covered) inside a `ViroARScene`. Caps render count, keeping covered voxels first. |
| `CoverageDemoScene.tsx` | `ViroARScene` for the demo. Runs `onCameraARHitTest`, picks the best surface hit, and forwards the world point via `onHitPoint` (throttled, gated on tracking == normal). |
| `CoverageDemoScreen.tsx` | Standalone AR harness (own tracker, independent of `SessionProvider`). Marks voxels via `observePoint` and shows live coverage % + tracking state. |
| `__tests__/CoverageTracker.test.ts` | Jest unit tests for the tracker (no React deps). |

## Two integration paths

1. **Capture gate (volume-based).** [features/capture/gates/CoverageGate.ts](../capture/gates/CoverageGate.ts) calls `tracker.peek(pose)` to gate keyframe acceptance. The shared tracker instance is owned by [providers/SessionProvider.tsx](../../providers/SessionProvider.tsx); `addKeyframe` commits the pose via `observe()`. Pose comes from the phone IMU via [capture/hooks/useDeviceMotionPose.ts](../capture/hooks/useDeviceMotionPose.ts) (rotation-only — `position` is always `[0,0,0]`), so the gate is orientation-based.
2. **Demo (surface-based).** `CoverageDemoScreen` uses AR hit-test points and `observePoint`, so cubes coat real geometry instead of filling the frustum volume. This path has true world positions.

## Voxel state machine

```
unseen   → not in map
partial  → in map, observationCount < CAPTURE_CONFIG.coverage.minObservations
covered  → in map, observationCount >= CAPTURE_CONFIG.coverage.minObservations
```

`coveragePercent` = covered / touched. It is a **quality** ratio (how well-observed the seen area is), not progress toward an unknown total volume — so it can decrease when the camera sweeps into fresh area.

## Notes / limitations

- `observe(pose)` marks the entire frustum cone (near `nearPlane` → far `frustumDepth`), not real surfaces. Use `observePoint` when AR depth/hit-test data is available.
- The demo requires a real ARKit/ARCore session; its hit-test path can't be exercised in Jest, only on-device.
