# VRoom Mobile App — Smart Mesh Capture Module

Build a React Native app that guides users during 3D capture via **continuous video recording** with **automatic keyframe extraction** — the user just walks around and the app picks the best frames.

---

## Review of Gemini's Suggested Methods

### Module 1: Blur Detection (Laplacian Variance) ✅ Math correct, library corrected

| Gemini Suggested | Corrected |
|---|---|
| `opencv.js` (WASM on canvas) | **`react-native-fast-opencv`** (native C++ via JSI, 10-50x faster) |
| `cv.imread('canvasElementId')` | VisionCamera frame processor → `OpenCV.frameBufferToMat()` |

### Module 2: Coverage Map (Voxel Grid) ⚠️ Right concept, wrong AR source

| Gemini Suggested | Corrected |
|---|---|
| **WebXR** for camera pose | **ViroReact** `getCameraOrientationAsync()` — WebXR needs a WebView sandbox, loses direct ARKit/ARCore access |
| **React Three Fiber** for overlay | **ViroReact `ViroARScene`** — single AR coordinate space |

### Module 3: Angle Diversity ✅ Correct as-is
Dot product + Euclidean distance gate — only swap WebXR pose source for ViroReact.

---

## Capture Model: Video → Auto Keyframes

Instead of manual shutter clicks, the app runs a **continuous capture loop**:

```
┌─────────────────────────────────────────────────────┐
│  Video Stream (30fps from VisionCamera)             │
│  ↓                                                  │
│  Frame Processor (runs every ~500ms)                │
│  ├── 1. Blur check: Laplacian variance > threshold? │
│  ├── 2. Angle diversity: far enough from last save? │
│  ├── 3. Coverage: does this view add new voxels?    │
│  ↓                                                  │
│  ALL PASS? → Auto-save frame + pose as keyframe     │
│  ANY FAIL? → Show guidance overlay to user          │
│             "Hold steady" / "Move sideways" / etc.  │
└─────────────────────────────────────────────────────┘
```

The user simply **presses Record, walks around the room, and stops**. The app handles frame selection.

---

## Architecture & Design Patterns

We will implement a **Feature-based Modular Architecture** combined with specific design patterns to keep the camera/AR performance high and business logic decoupled.

### 1. Project Folder Architecture (Feature-based)

```text
Mobile-APP/
├── App.tsx                       # Root entry point
├── src/
│   ├── app/                      # Global providers, navigation
│   ├── features/                 # Core business domains
│   │   ├── capture/              # Camera UI, Frame Processor, Keyframe extractor
│   │   ├── coverage/             # Voxel grid, AR rendering, Frustum math
│   │   └── export/               # Session packaging, File system, API upload
│   ├── shared/                   # Reusable cross-feature logic
│   │   ├── components/           # Generic UI (Buttons, HUDs, Banners)
│   │   ├── core/                 # Math utilities, standard Types
│   │   └── hooks/                # Custom React hooks
│   └── services/                 # External integrations
│       ├── opencv/               # JSI wrappers for react-native-fast-opencv
│       └── storage/              # Local file management
```

### 2. Design Patterns

**A. Strategy Pattern (Capture Gates)**
To determine if a frame becomes a keyframe, we implement an `ICaptureGate` interface (`evaluate(frame, pose) -> boolean`).
*   `BlurGate`: Checks Laplacian variance against a threshold.
*   `AngleGate`: Uses dot product and distance to prevent duplicate angles.
*   `CoverageGate`: Checks if the current view frustum observes empty voxels.
*   *Implementation*: The `KeyframeExtractor` loops through active gates. This obeys the Open/Closed Principle, allowing us to easily add new checks (e.g., brightness/exposure) later.

**B. Observer Pattern / Shared Values (UI Thread Sync)**
VisionCamera frame processors run thousands of times per minute on a background UI thread. Standard React `setState` would cause massive frame drops. 
*   *Implementation*: We use Reanimated **Shared Values** (an implementation of the Observer pattern). The Worklet thread mutates the shared value (`blurScore.value = score`), and the HUD overlay observes it directly, bypassing React's reconciliation cycle entirely.

**C. Facade Pattern (Complex Subsystems)**
We wrap the low-level `react-native-fast-opencv` and `react-viro` APIs behind simple facade services (e.g., `ImageQualityService.getBlurScore()`). 
*   *Implementation*: This isolates complex memory management (like calling `OpenCV.clearBuffers()`) and JSI interactions so UI components remain clean and declarative.

**D. Singleton / Hook Injection (Session State)**
The active capture session (saved camera poses, voxel matrix memory, frame count) must be accessed globally by the AR scene, the camera processor, and the HUD.
*   *Implementation*: We maintain this domain state in a singleton context provider (`SessionProvider`), ensuring the AR tracker and frame processor query the exact same memory references without prop-drilling.

### Dependencies

| Package | Purpose |
|---------|---------|
| `@reactvision/react-viro` | ARKit/ARCore (pose, anchors, AR rendering) |
| `react-native-vision-camera` | Camera + frame processors |
| `react-native-fast-opencv` | Native OpenCV via JSI |
| `react-native-worklets-core` | Worklet threading for frame processors |
| `vision-camera-resize-plugin` | Downscale frames before processing |

---

## Incremental Build Plan

### Build 1: Camera + Blur Detection (MVP)
- Bare React Native project, `react-native-vision-camera` setup
- [blurDetector.ts](file:///d:/Engineering/CUFE/GP2/VRoom/Mobile-APP/src/modules/blurDetector.ts): frame processor → grayscale → Laplacian → variance
- `CaptureHUD.tsx`: live blur score indicator + "Hold steady!" warning
- Simple record button — saves **all non-blurry frames** at ~2fps cadence
- Output: folder of images ready for COLMAP

### Build 2: AR Pose + Angle Diversity
- Add ViroReact, switch to `ViroARSceneNavigator`
- `angleDiversity.ts`: store poses, block saves if distance < 10cm AND cosine > 0.95
- `keyframeExtractor.ts`: combines blur + angle checks to gate auto-saves
- Save camera extrinsics (position, rotation, forward) alongside each keyframe

### Build 3: Coverage Map + Guided Prompts
- `coverageTracker.ts`: voxel grid + frustum intersection
- `VoxelOverlay.tsx`: AR voxel cubes (red/yellow/green)
- `keyframeExtractor.ts`: add coverage criterion — only save if frame adds new voxels
- Guided prompts ("Move to uncovered area", "Look upward", etc.)
- Coverage % bar in HUD

### Build 4: Session Export + Backend Upload  ← **NEXT**

Goal: close the capture → server loop so a finished session can actually produce a mesh.

**4a. Session summary screen** (`features/capture/SessionSummaryScreen.tsx`)
- Shown when `stopSession()` fires
- Stats: keyframe count, coverage %, duration, total MB on disk
- Thumbnail grid of captured keyframes (tap to inspect / delete bad ones)
- Buttons: "Upload to server" / "Discard session"

**4b. Session packaging** (`features/export/sessionPackager.ts`)
- Build `session.json` with `{ sessionId, startedAt, endedAt, deviceInfo, coveragePercent, keyframes: [{ filename, pose, blurScore, timestamp }] }`
- Zip all captured JPGs + `session.json` into one archive in the cache dir
- Use `react-native-zip-archive` (works with Expo dev build)

**4c. Backend upload** (`features/export/uploadService.ts`)
- POST multipart to `${API_BASE_URL}/sessions` with the zip
- Show progress (XHR upload progress events, since `fetch` doesn't expose them)
- Returns `{ sessionId, jobId }` — poll `/jobs/:jobId` for reconstruction status
- Configurable server URL in `.env` / app settings (default to LAN IP for dev)

**4d. Reconstruction status screen**
- Poll job state: `queued` → `colmap` → `gaussian-training` → `mesh-extraction` → `done`
- When done: fetch resulting `.glb`, save via `meshStorage`, route to AR viewer with the new mesh pre-selected

**4e. Backend endpoints** (minimal FastAPI server, separate task)
- `POST /sessions` — receive zip, unpack, enqueue job, return `jobId`
- `GET /jobs/:id` — return status + progress
- `GET /jobs/:id/result` — return final GLB
- Worker: run `object_isolation/run_pipeline.py` → `gstrain/trainer.py` → `Module4/extract_object_meshes.py`

### Build 5: AR Polish (after Build 4 is solid)
- **Multi-mesh placement** — currently only one mesh can be placed. Lift `meshSource`/`isMeshPlaced` into an array `placedMeshes: PlacedMesh[]` so users can stage a whole room
- Per-mesh selection / deletion in AR
- Save & restore AR scene layouts

---

## Verification (per build, physical device required)

| Build | Test |
|---|---|
| 1 | Record while steady → frames saved. Shake phone → "Hold steady!" + no frames saved |
| 2 | Stand still recording → stops saving after first frame. Walk sideways → resumes saving |
| 3 | Walk around room → coverage % climbs. Red voxels disappear. Guidance prompts appear for uncovered areas |
| 4 | Complete session → summary screen shows stats. Tap upload → zip lands on server, job runs, GLB returns and opens in AR viewer |
| 5 | Place 3 different meshes in one AR scene, move/rotate each independently |
