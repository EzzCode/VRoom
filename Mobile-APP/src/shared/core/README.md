# Shared Core

Framework-free domain primitives shared across features.

| File | Purpose |
|---|---|
| `types.ts` | Core types: `Vec3`, `CameraPose`, `Keyframe`, `GateResult`, `VoxelKey`, `MeshInfo`, `SessionMetadata`. |
| `math.ts` | Pure vector helpers: `dot`, `distance`, `length`, `normalize`, `subtract`, `cross`, `cosineSimilarity`. |

Keep this folder free of React, native, and IO dependencies so it stays unit-testable and importable from worklets.
