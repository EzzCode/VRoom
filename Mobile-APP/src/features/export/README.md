# Export Feature

Turns a finished capture session into a reconstruction job: previews keyframes,
uploads them to the backend, tracks the reconstruction, and downloads the
resulting mesh for AR viewing.

## Modules

| File | Purpose |
|---|---|
| `ExportScreen.tsx` | Session summary (keyframes, coverage %, duration, on-disk size), keyframe thumbnail grid, and Upload / Discard actions. |
| `sessionPackager.ts` | Builds `session.json` (`buildSessionManifest`), computes on-disk size (`getSessionDiskSize`), and cleans up JPEGs (`deleteSessionFiles`). Uses `expo-file-system/legacy`. |
| `uploadService.ts` | `uploadSession` (XHR multipart upload with progress), `fetchJobStatus` / `pollJobUntilComplete`, and `getJobResultUrl`. |
| `ReconstructionStatusScreen.tsx` | Polls the job through its stages (COLMAP → Gaussian training → mesh extraction), downloads the GLB into `documentDirectory/meshes/`, then navigates to the AR view. |
| `config.ts` | `API_BASE_URL` and `POLL_INTERVAL_MS`. |

## Flow

`CaptureScreen` (stop session) → `Export` → upload → `ReconstructionStatus` → `ARView`.

## Backend contract

The upload assumes a backend exposing:
- `POST /sessions` — accepts `sessionJson` (string field) + `images` (JPEG files); returns `{ sessionId, jobId }`.
- `GET /jobs/:id` — returns `{ jobId, state, progress?, message?, resultUrl? }`.
- `GET /jobs/:id/result` — the reconstructed GLB.

`API_BASE_URL` defaults to `http://127.0.0.1:8000` in [config.ts](./config.ts); for a USB device run `adb reverse tcp:8000 tcp:8000`.
