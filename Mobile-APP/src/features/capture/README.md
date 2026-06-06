# Capture Feature

Owns real-time camera capture (VisionCamera) and gated keyframe persistence.

## Layout

| Path | Purpose |
|---|---|
| `CaptureScreen.tsx` | UI orchestration: mounts the camera, runs the worklet frame processor (blur scoring), drives the capture loop, and renders the HUD. |
| `KeyframeExtractor.ts` | Strategy-pattern orchestrator: runs all registered gates; a frame is a keyframe only if **every** gate passes, else the first failure's reason is surfaced. |
| `gates/` | Capture gates implementing `ICaptureGate`: `BlurGate` (worklet sharpness), `AngleGate` (pose diversity), `CoverageGate` (voxel coverage). |
| `config/` | `captureConfig.ts` — sampling interval, blur threshold, HUD offset, angle-diversity and coverage tunables. |
| `hooks/` | `useDeviceMotionPose.ts` — fuses the phone IMU into a rotation-only `CameraPose` (position stays `[0,0,0]`) feeding `SessionProvider.setCurrentPose`. |
| `services/` | `captureStorage.ts` — saving captured photos to device storage. |
| `ar/` | AR-based capture (`ARCaptureScreen` + `ARCaptureScene`). Alternative path that runs on the Viro AR camera. |

## Two capture paths

| | `CaptureScreen` (default) | `ar/ARCaptureScreen` (beta) |
|---|---|---|
| Camera | VisionCamera | Viro AR (`ViroARSceneNavigator`) |
| Pose | IMU, rotation-only | Real 6DoF (`onCameraTransformUpdate`) |
| Coverage | frustum `observe(pose)` | surface `observePoint` via AR hit-test, live `VoxelOverlay` |
| Blur gate | real-time worklet | none (frame worklet can't run under Viro) |
| Keyframe capture | `camera.takePhoto()` | `arNavigator.takeScreenshot()` |

Both feed the same `SessionProvider` (keyframes, coverage %, export). The AR path
adds keyframes with `{ observeCoverage: false }` and commits coverage continuously
via `addCoveragePoint`. They are separate screens because VisionCamera and Viro
cannot own the camera simultaneously.

## Rules

- Keep camera frame processors worklet-safe.
- Avoid `any` in the capture hot path.
- New tunable values must live in `config/captureConfig.ts`.

## Notes

- The IMU pose feeding `AngleGate` / `CoverageGate` on the default path is rotation-only (no translation). See [coverage/README.md](../coverage/README.md).
- `BlurGate` runs inside the worklet, not the `KeyframeExtractor` pipeline.
- AR keyframes are PNG snapshots saved with a `.jpg` name; fine for content-sniffing decoders (COLMAP), but note the extension mismatch if a strict backend is added.
