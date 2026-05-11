# Capture Feature

This feature owns real-time camera capture and blur-gated frame persistence.

## Boundaries
- `CaptureScreen.tsx`: UI orchestration only.
- `gates/`: worklet-safe quality gates and frame scoring logic.
- `config/`: runtime constants for sampling, blur thresholds, and UI offsets.
- `services/`: persistence and device-side capture storage behavior.
- `types/`: bridge typings for native/OpenCV interop.

## Rules
- Keep camera frame processors worklet-safe.
- Avoid `any` in capture hot path.
- New tunable values must live in `config/`.
