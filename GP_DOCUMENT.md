# VRoom — Graduation Project Technical Documentation

---

## 4. System Design

### 4.1. System Overview

VRoom is an end-to-end pipeline that takes a casual video of an indoor room, reconstructs its 3D structure, isolates individual furniture objects, and delivers interactive, textured 3D meshes to a mobile AR application. The pipeline is fully automated from image capture to AR placement and is composed of six tightly-coupled modules, each handling a distinct stage of the processing chain.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         VRoom Pipeline                               │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Module 1 │───▶│ Module 2 │───▶│ Module 3 │───▶│ Module 4 │       │
│  │  Scene   │    │ Scene    │    │ Per-Obj  │    │  Mesh    │       │
│  │ Labeling │    │  3DGS    │    │ Gaussian │    │   Gen.   │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│       ▲                                                ▼             │
│  ┌──────────┐                               ┌──────────────────┐    │
│  │ Module 5 │                               │     Module 5     │    │
│  │  Mobile  │◀──────────────────────────────│  Mobile AR App   │    │
│  │  Capture │         .ply meshes           │  (AR Viewer)     │    │
│  └──────────┘                               └──────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 4.3. Module 1: Object Segmentation, Tracking & 3D Labeling

#### 4.3.1. Functional Description

This module is the foundation of the entire pipeline. It ingests a folder of raw room images and outputs a semantically labeled 3D sparse point cloud where every point carries the identity of the object it belongs to. The module is fully automated and requires no manual annotation.

The module takes as input a flat directory of RGB images (JPG/PNG) of an indoor scene. It performs Structure-from-Motion reconstruction using COLMAP to recover camera poses and a sparse point cloud, applies the SAM3 open-vocabulary segmentation model to detect and segment furniture objects in every image using text prompts, tracks those object masks consistently across frames using a Kalman-filter + Hungarian-algorithm tracker, and finally projects the 3D sparse points back onto all 2D frames to assign each 3D point a majority-voted object label. The output is a labeled `.ply` file of the sparse reconstruction, with per-point RGB color and integer object ID, alongside per-frame ID map images used by downstream modules.

#### 4.3.2. Modular Decomposition

The module operates as a sequential four-stage pipeline:

**Stage 1 — Structure-from-Motion (COLMAP)**

COLMAP is invoked automatically to perform feature extraction (SIFT), sequential or exhaustive feature matching, and sparse bundle-adjustment reconstruction. The output is a database and a `sparse/0/` folder containing `cameras.bin`, `images.bin`, and `points3D.bin` — the camera intrinsics and extrinsics for every registered image and the 3D sparse point positions.

This stage is configurable: the camera model (PINHOLE, OPENCV, SIMPLE_RADIAL), the matcher type (sequential for video, exhaustive for unordered sets), and whether to force a re-run are all user-controllable.

**Stage 2 — Open-Vocabulary Segmentation (SAM3)**

SAM3 (`SAM3TextSegmenter`) is applied to every image in the set. The model is loaded from an Ultralytics checkpoint and run in text-prompted semantic mode. Default prompts are furniture categories: *chair, table, sofa, bed, desk, cabinet*. Each prompt is used to produce raw boolean foreground masks.

Raw masks are then refined by `mask_processor.py` which applies five sequential spatial filters:
- **Background / border-touch filter**: discard masks that cover an excessive fraction of the image area, or whose boundary touches too many image edges — these are likely background or walls.
- **Containment merge**: if one mask is almost entirely contained within another (IoU threshold), they are merged into one.
- **Proximity + color merge**: nearby masks that share similar mean HSV color are merged, addressing the case where one object is split by SAM into two nearby regions.
- **Connected-component split**: any remaining mask with multiple disconnected components is split into separate masks.
- **Minimum area gate**: masks below a pixel-count threshold are discarded.

Each processed frame produces a compressed `.npz` archive of boolean masks and a debug overlay PNG.

**Stage 3 — Multi-Object Tracking**

`object_tracker.py` reads the per-frame mask archives and links them across frames to maintain stable integer object IDs. The tracker uses a Kalman filter to predict each tracked object's bounding-box position in the next frame, and the Hungarian algorithm to optimally assign new detections to existing tracks based on a combined cost matrix of IoU overlap, bounding-box distance, and color similarity (weighted by configurable α, β, γ parameters).

Tracks must pass a temporal consensus check before they are confirmed as real objects — a track must be continuously observed for a minimum number of consecutive frames before being assigned a stable ID. Lost tracks are held for a configurable number of frames before being discarded, to handle brief occlusions.

The output is a `tracked/id_maps/` folder containing one 16-bit PNG per image, where each pixel value equals the stable integer track ID of the object at that location (0 = background).

**Stage 4 — 3D Voting**

`vote.py` reads the COLMAP sparse point cloud and the per-frame ID maps. For each 3D point, it projects it onto every camera image in which it was observed (using the COLMAP track data), reads the ID map value at the projected pixel, and collects votes. The majority vote determines the final label of the 3D point. Points with fewer than a minimum number of observations, or where votes are inconclusive, are labeled as background.

A post-processing alias-merging step resolves the case where one physical object received two different tracker IDs across disjoint trajectory windows (e.g., an object not seen for a long time). Two tracker IDs are merged if their 3D labeled point sets overlap spatially, using a nearest-neighbor graph and connected-components analysis.

The output is a `points3D_labeled.ply` file.

#### 4.3.3. Design Constraints

- **COLMAP quality dependence**: The entire pipeline relies on a successful COLMAP reconstruction. Scenes with repetitive textures (e.g., plain white walls) or severe motion blur may produce incomplete reconstructions with inaccurate camera poses, which degrade every downstream stage.
- **SAM3 model dependency**: The segmentation stage requires a GPU-resident SAM3 checkpoint (~1–2 GB). CPU inference is supported but is significantly slower.
- **Tracking ID fragmentation**: Objects that are occluded for many frames may receive multiple distinct tracker IDs, which the alias-merging step attempts to resolve, but cannot guarantee perfect consolidation in all cases.
- **Open-vocabulary coverage**: SAM3's performance is tied to the quality of the text prompts. Objects not captured by the prompt vocabulary will not be segmented.

> **Status**: Fully implemented and tested on indoor room scenes.

---

### 4.4. Module 2: Scene Reconstruction via 3D Gaussian Splatting

#### 4.4.1. Functional Description

This module trains a full-scene 3D Gaussian Splatting (3DGS) model from the registered images and labeled sparse point cloud produced by Module 1. The result is a compact, differentiable 3D representation of the entire room — a Scaffold-GS model — where every Gaussian primitive carries a semantic object-label attribute, enabling all downstream modules to isolate specific objects by label.

The input is the COLMAP sparse reconstruction (camera poses + labeled point cloud), optionally accompanied by depth maps and per-pixel object masks. The output is a trained model checkpoint (`.ply` + MLP weights) that can be rendered from any novel viewpoint.

#### 4.4.2. Modular Decomposition

**Scene Data Loading**

The training scene is built from the COLMAP output. Camera intrinsics and extrinsics are read, and training images are loaded alongside optional depth maps and semantic masks. The labeled sparse point cloud seeds the initial Gaussian positions and colors.

**Scaffold-GS Model**

The Gaussian model is a Scaffold-GS variant (`GaussianModel` in `gstrain/vroom_core/models/`). Rather than representing every Gaussian with individual attributes, Scaffold-GS organizes Gaussians hierarchically around anchor points in a voxel grid. Each anchor owns a set of neural Gaussians whose properties (opacity, color, scale, rotation) are decoded from compact MLP networks conditioned on the viewing direction and a feature vector. This dramatically reduces memory and enables finer detail with fewer primitives.

Gaussians inherit the integer object label of the nearest labeled point in the initial point cloud, giving the model per-Gaussian semantic identity from the start of training.

**Training Loop**

The `TrainingOrchestrator` runs a standard differentiable rendering loop:
1. Sample a training camera.
2. Render the scene to an image using the differentiable rasterizer (`diff-surfel-rasterization`).
3. Compute the photometric loss: a weighted combination of L1 pixel loss and SSIM.
4. Optionally add depth supervision (L1 loss against the depth maps, ramped in after a configurable number of iterations).
5. Add regularization terms: distortion loss, normal consistency loss, opacity entropy loss.
6. Backpropagate through the rasterizer and update all MLP and Gaussian parameters.

Adaptive densification is applied periodically: Gaussians with high accumulated positional gradient are cloned or split to better capture fine detail, while low-opacity Gaussians are pruned.

**Loss Engine**

`loss_engine.py` centralizes all loss computation. Losses are individually weighted by configurable λ parameters (λ_dssim, λ_dreg, λ_object_loss, etc.) and can be enabled or disabled independently, enabling ablation studies.

#### 4.4.3. Design Constraints

- **CUDA requirement**: Training requires a CUDA-capable GPU. The rasterizer (`diff-surfel-rasterization`) is a compiled CUDA extension and must be built before use (`make clean build`).
- **Label noise propagation**: Erroneous labels from Module 1 are inherited by the Gaussian model. Noise in the labeled point cloud will be partially corrected during training by the photometric supervision, but cannot be fully eliminated.
- **Training time**: Full scene training takes approximately 30,000 iterations. On a mid-range GPU, this takes between 30 and 90 minutes depending on scene complexity and image resolution.

> **Status**: Fully implemented. The model is built on top of ObjectGS / Scaffold-GS and extended with semantic label support and depth supervision.

---

### 4.5. Module 3: Per-Object Gaussian Refinement

#### 4.5.1. Functional Description

This module takes the full-scene Gaussian model from Module 2 and produces a separate, standalone, high-quality Gaussian model for each individual object in the scene. The key challenge is that any given object is only partially visible from the training cameras — some of its surfaces face away from all captured viewpoints and therefore have no photometric supervision. This module addresses the problem by hallucinating novel views of the object from unseen angles using a video diffusion prior (SV3D), then using those hallucinated views as additional supervision during per-object training.

The input is the trained scene model checkpoint, the scene images directory, and optionally the Module 1 ID maps. The output is a folder per object containing a trained per-object Gaussian checkpoint and comparison renders.

#### 4.5.2. Modular Decomposition

The module operates as a four-stage pipeline per object:

**Stage 1 — Object Extraction and Scope Computation**

Given the scene model and an object label ID, `view_selection.run_extraction()` iterates over all training cameras. For each camera, the scene model is rendered with only the target object's Gaussians active (by filtering on the semantic label). The resulting alpha mask indicates which pixels belong to the object in that viewpoint.

A hybrid mask is then computed by intersecting the model-rendered alpha mask with the Module 1 ID map for that frame (if available). This hybrid mask is more precise than either source alone — the 3DGS render provides 3D-consistent boundaries while the Module 1 map provides pixel-accurate 2D boundaries.

From the set of per-camera masks, the module computes the object's 3D bounding sphere (center and radius) in world space using the masked depth maps, and builds the camera extrinsics relative to this object center (`ObjectFrame`).

**Stage 2 — Best-View Frame Scoring**

`view_selection.run_scoring()` scores every extracted frame on five criteria:
- **Frontality** (weight 0.35): prefer cameras roughly facing the object's centroid.
- **Coverage** (weight 0.20): prefer frames where the object occupies a significant fraction of the image, up to a ceiling.
- **Sharpness** (weight 0.20): Laplacian variance of the masked object region, rank-normalized across all frames.
- **Exposure** (weight 0.10): penalize over- or under-exposed frames.
- **Occlusion** (weight 0.15): higher fraction of the hybrid mask that is unoccluded.

The top-K frames by composite score are selected as conditioning views for the hallucination stage.

**Stage 3 — Novel View Hallucination (SV3D)**

`hallucination.run_hallucination()` uses the SV3D video diffusion model to synthesize a 360° orbit of novel view images around the object. The best-scored frame is cropped tightly to the object, alpha-composited onto a neutral background, and used as the conditioning image for SV3D. SV3D generates a temporally consistent multi-view sequence at user-specified elevation angles.

Each generated frame is post-processed: a foreground alpha mask is estimated from the white-background SV3D output using HSV thresholding and morphological operations. The frames are then normalized and stored as RGBA images alongside inferred camera poses on a spherical orbit around the object.

**Stage 4 — Per-Object Gaussian Training**

`object_refiner.trainer.train_object()` initializes a fresh Gaussian model seeded from the labeled COLMAP point cloud subset for the target object. If too few labeled points exist, the initialization is upsampled via neighbor interpolation.

The training loop jointly supervises the model on:
- **Real views**: photometric RGB loss (L1 + SSIM) on the actual training cameras, masked to the object region.
- **Hallucinated views**: photometric RGB loss on the SV3D-generated novel views. These are downweighted by a configurable `hallucination_weight` to account for their lower fidelity compared to real images.
- **Depth supervision**: L1 depth loss (ramped in after a warm-up period) using depth maps extracted from the scene model, to maintain geometric consistency.

After training, orbit comparison renders (before/after) are saved to the output directory for visual quality assessment. The model is exported in ObjectGS-compatible layout.

#### 4.5.3. Design Constraints

- **SV3D external dependency**: The hallucination stage requires the SV3D model weights (Stable Video Diffusion) loaded via the `diffusers` library. This adds a significant GPU memory requirement (~10–12 GB) and a non-trivial inference time per object (~2–5 minutes on an A100). The model can be bypassed (`--reuse_sv3d` flag) if cached hallucination output already exists.
- **View coverage assumption**: The method assumes that at least one reasonable front-facing view of the object is present in the training set. If all views are occluded or extremely oblique, the hallucination conditioning will be poor.
- **Hallucination fidelity**: SV3D is a generative model and may hallucinate plausible but incorrect geometry or texture on truly unobserved surfaces. The real-view supervision acts as a regularizer but cannot fully override this.
- **Per-object training cost**: Training one object takes approximately 1,000–12,000 iterations (~2–15 minutes per object on a mid-range GPU). A scene with ten objects may take several hours total.

> **Status**: Fully implemented. SV3D integration is functional but requires separate model weight download. The training and extraction stages are stable. The comparison render pipeline is complete.

---

### 4.6. Module 4: (Placeholder — object_isolation integration layer)

> **Status**: The `object_isolation/` directory contains an earlier integration layer for orchestrating per-object training directly from the scene model. This has been superseded by the more complete object_refiner pipeline (Module 3). The remaining utilities (`run_pipeline.py`, `run_training.py`) are retained for compatibility but are not part of the primary pipeline.

---

### 4.7. Module 5: Mesh Generation

#### 4.7.1. Functional Description

This module is the final computational step, transforming the trained per-object Gaussian models from Module 3 into polygonal surface meshes that can be used in game engines, AR/VR platforms, and standard 3D software such as MeshLab or Blender. A separate colored `.ply` mesh file is produced for each individual object in the scene.

The module takes as input RGB renders of the Gaussian scene, semantic segmentation maps that separate objects in the scene, camera parameters, and depth maps extracted from the trained Gaussian model. Every object is then isolated based on its semantic label. For each object, TSDF (Truncated Signed Distance Function) fusion is performed to build a 3D volumetric grid, followed by the Marching Cubes algorithm to extract the final surface mesh. Consequently, a standalone `.ply` file is generated for each object. All `.ply` files, or a subset of them, can be opened together in software like MeshLab to visualize the full scene.

#### 4.7.2. Modular Decomposition

The module operates as a sequential pipeline of six sub-stages:

**Sub-module 1 — Data Ingestion**

All inputs are loaded into memory from a structured input folder:
- `renders/` — RGB images of the Gaussian scene, one per camera, in PNG format (named `00000.png`, `00001.png`, …).
- `raw_depth/` — Depth maps of each camera as `.npy` files (NumPy float arrays).
- `semantic/` — Semantic segmentation maps, one per camera, in PNG format, where pixel values are integer object label IDs.
- `cameras.json` — Camera metadata array containing, for each camera: camera-to-world rotation matrix, position vector, focal lengths (fx, fy), and image dimensions.

All data sources are indexed in lock-step so that the depth map, RGB image, semantic map, and camera parameters for a given viewpoint are always loaded together.

**Sub-module 2 — Object Isolation and Bounding Box**

For each unique object label found in the semantic maps, the module performs the following steps:

1. **Pixel count filter**: All pixels belonging to the object are summed across all views. If the total falls below `MIN_PIXELS` (default: 10,000), the object is skipped — this eliminates noise labels and very small objects irrelevant to the furniture-scale use case.

2. **Depth truncation**: A per-object depth truncation value is computed to exclude pixels that may erroneously belong to the object in the semantic mask while representing a far background surface. The Nth percentile of all object-masked depth values is computed (default: 99th percentile), and then multiplied by a margin factor (default: 1.1). This removes hard outliers while preserving valid far-edge points with a small safety buffer.

3. **3D unprojection and bounding box**: All valid object pixels across all cameras are unprojected into 3D world space using the depth map and camera intrinsics/extrinsics. The resulting 3D point cloud is clipped at the `BBOX_CLIP` percentile on each axis (default: 2%) to remove extreme outliers, and a padded axis-aligned bounding box is computed with `PADDING` fractional margin (default: 22%). This bounding box defines the TSDF grid volume.

**Sub-module 3 — GPU-Accelerated TSDF Fusion**

`generate_sdf.fuse_tsdf()` accumulates the signed distance function across all cameras into a 3D voxel grid of resolution N³ (default: 128³):

For each camera, the module:
1. Projects all N³ voxel centers into the camera image plane using the intrinsics and extrinsics matrices.
2. Filters out voxels that project outside image boundaries or behind the camera.
3. Bilinearly interpolates the depth map at each projected pixel to obtain the surface depth at that voxel's ray.
4. Computes the signed distance: `sdf = surface_depth − voxel_depth`. Positive means the voxel is in front of the surface; negative means behind.
5. Clamps the SDF to the truncation margin `trunc_margin = voxel_size × TRUNC_FACTOR` (default: 5.0 voxel diameters).
6. Accumulates the SDF value and a weight into the voxel grid (only for voxels that meet the depth truncation criterion).

Voxels observed by fewer than `MIN_OBS` cameras (default: 3) are masked out as insufficiently constrained. The final TSDF grid is the weighted average of all camera contributions.

RGB color accumulation is performed in parallel: the RGB value at each projected pixel is bilinearly interpolated and accumulated per-voxel, yielding a colored TSDF grid.

The entire computation is vectorized using PyTorch GPU tensors — no per-voxel Python loops. A shared world-point matrix (N³ × 4 homogeneous coordinates) is computed once and reused across all cameras.

**Sub-module 4 — Surface Smoothing**

Before mesh extraction, a Gaussian smoothing filter (`scipy.ndimage.gaussian_filter`) is applied to the TSDF grid with a configurable sigma (default: 0.8 voxels). This suppresses high-frequency noise in the signed distance field caused by depth map quantization, TSDF truncation artifacts, and slightly misaligned camera poses, resulting in a smoother reconstructed surface.

**Sub-module 5 — Marching Cubes**

`marching_cubes.run_marching_cubes()` applies the Marching Cubes algorithm to the smoothed TSDF grid to extract an isosurface at the zero level-set, which corresponds to the reconstructed object surface.

The implementation is fully vectorized using NumPy with no per-voxel Python loops:
1. All (N−1)³ voxel cubes are processed simultaneously by extracting the 8 corner SDF values using array slicing.
2. Each voxel's 8-bit Marching Cubes index is computed in one pass using vectorized bitwise OR.
3. Active voxels (where the surface passes through) are identified in bulk.
4. For each of the 12 possible cube edges, all voxels where that edge is intersected are found simultaneously, and the surface vertex positions are batch-interpolated.
5. Triangle assembly uses the standard 256-case lookup table (`mc_tables.py`) but processes all voxels sharing the same case type together.

Vertex colors are interpolated from the TSDF color grid at each surface vertex position, producing a vertex-colored mesh.

**Sub-module 6 — Small Component Removal and PLY Export**

The raw mesh may contain small disconnected fragments caused by segmentation noise or sensor artifacts. `utils.remove_small_components()` identifies connected components of the mesh by building a vertex adjacency graph and removes any component whose vertex count is smaller than `MIN_COMPONENT` fraction (default: 5%) of the largest component.

`export_ply.export_ply_binary()` writes the final mesh to a binary `.ply` file with interleaved vertex position (3× float32) and vertex color (3× uint8) data, followed by face indices (1× uint8 count + 3× int32). The binary format is chosen over ASCII for compactness (roughly 4–6× smaller files).

#### 4.7.3. Design Constraints

- **Noise Amplification**: If the segmentation from Module 1 leaves noise around an object boundary (e.g., partial pixels from a neighboring wall), the TSDF will accumulate inconsistent depth values at those pixels, causing bumps or false surfaces in the mesh.
- **Resolution–Speed Trade-off**: The TSDF grid resolution N is the dominant cost parameter. Increasing N from 128 to 256 multiplies voxel count by 8, which multiplies both GPU memory usage and computation time by approximately 8×. For large objects such as sofas and beds, higher resolutions (256–512) may be needed to capture fine detail, at significant cost.
- **Depth Map Accuracy**: Depth maps are extracted from the trained Gaussian model, not from a physical depth sensor. Depth quality depends directly on the quality of the Gaussian training and may be imprecise in textureless regions.
- **Single-Object Bounding Box Assumption**: Each object is reconstructed independently inside its own bounding box. Objects that are very close together may have overlapping bounding boxes, and depth values from one object may contaminate the TSDF of another.

> **Status**: Fully implemented and tested. GPU-accelerated TSDF fusion, vectorized Marching Cubes, and binary PLY export are all complete. The pipeline produces watertight, vertex-colored meshes for indoor furniture objects.

---

### 4.8. Module 6: Mobile AR Application

#### 4.8.1. Functional Description

The mobile application provides the user-facing interface for the VRoom system. It serves two roles: (1) guided image capture at the beginning of the pipeline, where the user scans their room and the app intelligently selects high-quality keyframes to upload; and (2) AR visualization at the end of the pipeline, where the generated object meshes are downloaded and placed interactively inside the user's physical space using the device camera and plane detection.

The application is built in React Native (TypeScript) and targets iOS and Android. It uses `react-native-vision-camera` for real-time camera access, `react-native-fast-opencv` for on-device image processing in camera worklets, and `@reactvision/react-viro` for the AR rendering layer.

#### 4.8.2. Modular Decomposition

**Feature: Scene Capture**

`CaptureScreen.tsx` orchestrates the room scanning session. The user presses record to begin a session and the app continuously evaluates the live camera feed through a chain of quality gates implemented in the `gates/` sub-directory. Each gate runs inside a VisionCamera frame processor worklet on a dedicated native thread, ensuring the main UI thread is never blocked.

Gates currently implemented:
- **Blur Gate** (`BlurGate.ts`): Computes the Laplacian variance of a downscaled grayscale version of the frame using `react-native-fast-opencv`. A frame whose variance falls below a threshold is rejected as too blurry. The pipeline is: BGR buffer → Mat → grayscale → Laplacian → meanStdDev → variance.
- **Angle Gate** (`AngleGate.ts`): Evaluates the device's physical tilt angle (obtained from AR pose or device IMU) to reject frames taken at extreme angles unlikely to reconstruct well.

The `KeyframeExtractor` class implements a Strategy Pattern over the gate chain: a frame becomes a keyframe only if all registered gates pass. If any gate fails, its guidance message is surfaced to the HUD so the user knows to move the camera.

Accepted keyframes are saved to device storage via `captureStorage.ts`. The app enforces a minimum time interval between captures (configurable, default: 1200 ms) to prevent redundant near-duplicate frames.

**Feature: Mesh Gallery**

`MeshGallery.tsx` provides a library screen showing all available `.ply` meshes, either imported from the file system or bundled with the app as sample assets. Users can browse, search, and import new mesh files via the native file picker (`importMeshFromFilePicker`). Each mesh card shows the mesh name, file size, and format.

**Feature: AR Viewer**

`ARViewScreen.tsx` implements the AR placement experience using `ViroARSceneNavigator` and a custom `ARMeshScene`. The user selects a mesh from the gallery and is taken to a live AR camera view. On first launch, a reticle (`ARReticle`) is shown to guide the user to point at a flat surface for plane detection.

Interaction modes:
- **Place**: Tap detected plane to anchor the mesh at that world position.
- **Move**: Drag the placed mesh to reposition it on the detected plane.
- **Rotate**: Rotate the mesh around its vertical axis.
- **Scale**: Pinch to scale the mesh up or down.

AR tracking state is monitored continuously and a HUD indicator (`AROverlayUI`) reflects whether the AR engine has limited or normal world tracking. Screenshots of the AR scene can be taken and saved to the device photo library.

#### 4.8.3. Design Constraints

- **AR engine dependency**: The AR experience relies on `@reactvision/react-viro` which wraps ARKit (iOS) and ARCore (Android). AR plane detection quality is device- and environment-dependent and degrades in low-light or textureless environments.
- **Mesh format compatibility**: The AR viewer currently supports `.ply` meshes (vertex-colored). Support for `.obj` and `.glb` formats is possible through `react-viro`'s `Viro3DObject` component but requires server-side format conversion from `.ply`.
- **Export and Coverage features**: The `export/` and `coverage/` feature directories are reserved scaffolding and are not yet implemented. Export (for sharing scans externally) and coverage analysis (for guiding the user to scan uncovered regions) are planned for future work.
- **On-device processing**: The blur and angle gates run entirely on-device in real time. No processing is offloaded to the server during capture — only the selected keyframe images are uploaded at the end of the session.
- **Network upload**: The interface between the mobile app and the server-side pipeline (Modules 1–5) is a planned integration point. The current app stores keyframes locally and would require a backend upload service to complete the end-to-end flow.

> **Status**: Core capture pipeline (blur gating, angle gating, keyframe extraction, session management) is fully implemented. The AR viewer (mesh placement, move, rotate, scale, screenshot) is fully implemented. The mesh gallery (browse, import) is fully implemented. The Export and Coverage features are scaffolded but not implemented.

---

## 4.9. End-to-End Pipeline Summary

| Module | Folder | Status | Key Technology |
|--------|--------|--------|----------------|
| Module 1a: Structure-from-Motion | `sfm/` | ✅ Complete | COLMAP reconstruction |
| Module 1b: Masks & Tracking | `masks_and_tracking/` | ✅ Complete | SAM3 (Ultralytics), Kalman + Hungarian tracking, 3D majority voting |
| Module 2: Scene 3D Gaussian Splatting | `gstrain/` | ✅ Complete | Scaffold-GS, differentiable rasterizer (CUDA), depth supervision |
| Module 3: Per-Object Gaussian Refinement | `object_refiner/` | ✅ Complete (SV3D weights required separately) | SV3D (video diffusion), COLMAP init, Gaussian Splatting fine-tuning |
| Module 4: Integration Layer | `object_isolation/` | ⚠️ Superseded by Module 3 | — |
| Module 5: Mesh Generation | `Module4/` | ✅ Complete | GPU TSDF fusion (PyTorch), Marching Cubes (NumPy), binary PLY export |
| Module 6: Mobile AR Application | `Mobile-APP/` | 🔶 Partial (Export & Coverage TBD) | React Native, VisionCamera, react-native-fast-opencv, ViroAR |

### Data Flow Between Modules

```
Module 1 Output:
  sparse/0/points3D_labeled.ply    ─────────────────────▶ Module 2 input
  tracked/id_maps/*.png            ─────────────────────▶ Module 3 input

Module 2 Output:
  <model_path>/point_cloud/*.ply   ─────────────────────▶ Module 3 input
  <model_path>/config.yaml         ─────────────────────▶ Module 3 input

Module 3 Output (per object):
  <output>/point_cloud.ply         ─────────────────────▶ Module 5 input
  <output>/renders/*.png           ─────────────────────▶ Module 5 input
  <output>/depth/*.npy             ─────────────────────▶ Module 5 input
  <output>/semantic/*.png          ─────────────────────▶ Module 5 input
  <output>/cameras.json            ─────────────────────▶ Module 5 input

Module 5 Output (per object):
  objects/obj_<id>.ply             ─────────────────────▶ Mobile App (AR Viewer)
```
