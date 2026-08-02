Got it. This is the implementation plan for GPT‑5.6 xhigh to execute later. No repository changes are included now.

# ScanLan implementation plan

## Objective

Extend ScanLan so that:

* Every RGB-D camera records synchronized native-resolution RGB alongside the existing depth-aligned RGB.
* Textured meshes use native-resolution RGB.
* Kinect v2, Azure Kinect and Femto scans additionally export depth-constrained Gaussian splats.
* Photo folders and videos can produce RGB-only Gaussian splats.
* Point cloud, textured mesh and Gaussian splat remain independent artifacts sharing one coordinate system.

## Locked architectural decisions

1. Preserve the existing RGB-D reconstruction path and outputs.
2. Keep depth-aligned RGB for odometry, TSDF and live preview.
3. Use native-resolution RGB for mesh textures and splat training.
4. Use Open3D’s optimized poses for RGB-D splats; do not rerun COLMAP on sensor captures.
5. Use FFmpeg and COLMAP for photos/videos. Nerfstudio already uses this combination for custom media. [Nerfstudio custom datasets](https://docs.nerf.studio/quickstart/custom_dataset.html)
6. Use Nerfstudio/Splatfacto with a ScanLan depth-supervision extension backed by gsplat.
7. Package splat support as an optional isolated Python environment, not inside the current Open3D PyInstaller executable.
8. Export canonical 3DGS PLY plus metadata. Do not initially bake arbitrary viewer transforms into the Gaussian data.
9. Maintain backward compatibility with existing project and phase archives.

## 1. Lock conventions and fixtures

Before structural changes, establish golden fixtures:

* One short Kinect v2 capture.
* One Azure/Femto capture with IMU.
* One 40–80-photo room dataset.
* One short handheld video.
* One expected Unity orientation fixture containing labeled X/Y/Z geometry.

Define the canonical project frame:

* Right-handed
* Metres
* `worldFromCamera` matrices
* Row-major matrices on disk
* Explicit camera convention recorded in every artifact manifest

Select and document the expected 3DGS PLY properties:

```text
x y z
nx ny nz
f_dc_0..2
f_rest_*
opacity
scale_0..2
rot_0..3
```

Acceptance gate: a known Splatfacto export opens correctly in the intended Unity renderer with verified position, scale and handedness.

## 2. Add backward-compatible schemas

Modify:

* `src-tauri/src/models.rs`
* `src/lib/types.ts`
* `src-tauri/src/storage.rs`
* `worker/scanlan/io.py`
* `docs/scan-format.md`

Bump the project schema and extend it additively:

```json
{
  "mediaSources": [],
  "artifacts": {
    "pointCloud": null,
    "texturedMesh": null,
    "gaussianSplat": null
  },
  "activeJob": null
}
```

Keep existing `outputPath`, `meshOutputPath`, and related fields populated until the UI has fully migrated.

Bump phase schema to v3 and add optional RGB calibration:

```json
{
  "rgbCamera": {
    "width": 3840,
    "height": 2160,
    "fx": 0,
    "fy": 0,
    "cx": 0,
    "cy": 0,
    "model": "brown_conrady",
    "distortion": []
  },
  "rgbFromDepth": [16],
  "sourceRgb": {
    "format": "jpeg",
    "quality": 92,
    "nativeResolution": true
  }
}
```

`rgbFromDepth` must explicitly map depth-camera coordinates into RGB-camera coordinates.

Append optional columns to `frames.csv`:

```text
rgb_path,rgb_timestamp_us
```

Old archives without these fields continue using aligned `color_path` as a lower-resolution fallback.

## 3. Introduce durable artifact jobs

The current Rust orchestration waits for one blocking `reconstruct` command in [commands.rs](https://github.com/benkuper/ScanLan/blob/main/src-tauri/src/commands.rs). Splat training requires a persistent job lifecycle.

Add:

* `src-tauri/src/jobs.rs`
* `outputs/jobs/<job-id>.json`
* Tauri commands for start, status, cancel and resume

A job record should include:

```json
{
  "id": "...",
  "pipeline": "rgbd_reconstruction",
  "targets": ["pointCloud", "texturedMesh", "gaussianSplat"],
  "stage": "splat_training",
  "progress": 0.42,
  "iteration": 12600,
  "totalIterations": 30000,
  "loss": 0.018,
  "etaSeconds": 420,
  "status": "running"
}
```

Requirements:

* Return the job ID immediately.
* Persist stdout/stderr logs.
* Gracefully cancel workers.
* Resume splat checkpoints when source fingerprints match.
* Never run Open3D CUDA fusion and gsplat training concurrently.
* Preserve the existing progress UI fields during migration.

## 4. Capture native-resolution RGB

Modify both:

* `native/kinect-capture/src/main.cpp`
* `native/modern-capture/src/main.cpp`

Retain the existing files:

```text
depth/000000.u16
color/000000.rgb
```

Add:

```text
rgb/000000.jpg
```

Capture requirements:

* Store the sensor’s native RGB resolution by default.
* Associate each RGB frame with its device timestamp.
* Record RGB intrinsics, distortion and depth-to-RGB extrinsics.
* Preserve the existing aligned RGB path unchanged.
* Encode JPEG asynchronously through a bounded writer queue so disk compression cannot stall depth capture.
* Record RGB drops separately; dropping a source-RGB frame must not invalidate the corresponding depth frame.
* Use a configurable JPEG quality, defaulting around 92.
* Allow an optional maximum RGB dimension, but default to native resolution.

Use Windows Imaging Component or an equivalent common native helper shared by both capture workers.

Acceptance gate for every camera:

* Depth/aligned RGB capture remains unchanged.
* Native RGB opens successfully.
* Timestamp pairing is valid.
* A projected calibration target lands within an agreed pixel tolerance.

## 5. Create the canonical posed-frame dataset

Add:

* `worker/scanlan/dataset.py`
* `worker/scanlan/calibration.py`

Extend:

* `worker/scanlan/open3d_engine.py`
* `worker/scanlan/reconstruct.py`

Generate a fingerprinted dataset after final pose-graph optimization:

```text
outputs/cache/datasets/<fingerprint>/
├── dataset.json
├── images/
├── depths/
├── masks/
└── initialization.ply
```

Each record contains:

```json
{
  "image": "images/000042.jpg",
  "depth": "depths/000042.png",
  "depthMask": "masks/000042.png",
  "worldFromRgbCamera": [16],
  "intrinsics": {},
  "timestampUs": 0,
  "phaseId": "...",
  "metric": true
}
```

For RGB-D frames:

[
worldFromRgb = worldFromDepth \cdot depthFromRgb
]

where:

[
depthFromRgb = (rgbFromDepth)^{-1}
]

Create RGB-view depth maps by back-projecting depth pixels, transforming them into the RGB camera, projecting them at native RGB resolution and resolving collisions with a nearest-depth z-buffer.

Do not derive training poses from the existing viewer-oriented `camera-poses.json`; export them from `PosedFrame` before display-axis conversion.

## 6. Upgrade textured-mesh generation

Refactor [mesh.py](https://github.com/benkuper/ScanLan/blob/main/worker/scanlan/mesh.py) to consume the canonical dataset.

For every depth-derived vertex:

1. Back-project the depth pixel.
2. Transform it using `rgbFromDepth`.
3. Project it with RGB intrinsics and distortion.
4. Reject projections behind the RGB camera or outside the image.
5. Compare projected depth against an RGB-view z-buffer.
6. Generate atlas UVs from the accepted native-resolution RGB coordinate.

Implement in two passes:

### First pass

Use the synchronized RGB frame paired with each depth keyframe. This gives correct full-resolution projection with minimal temporal mismatch.

### Quality pass

Select the best RGB view per triangle using:

* Visibility
* Surface-to-camera angle
* Distance
* Projected texel density
* Occlusion
* Exposure consistency

Improve atlas generation:

* Crop to used image regions.
* Add padding around atlas islands.
* Support configurable 8K/16K atlases.
* Downscale only when atlas capacity requires it.
* Add exposure compensation and seam feathering.
* Preserve the current OBJ/MTL/PNG export contract.

Acceptance gate:

* New captures produce visibly sharper textures.
* UVs remain within atlas bounds.
* Occluded background color does not bleed across foreground edges.
* Old archives still use the existing aligned-RGB fallback.

## 7. Add the optional splat runtime

Create a separate package:

```text
splat-worker/
├── pyproject.toml
└── scanlan_splat/
    ├── cli.py
    ├── dataset.py
    ├── train.py
    ├── depth_loss.py
    └── export.py
```

Add:

* `scripts/build-splat-runtime.ps1`
* `npm run prepare:splat`
* Runtime detection in `RuntimeInfo`

Pin tested versions of:

* Python
* PyTorch/CUDA
* Nerfstudio
* gsplat
* COLMAP
* FFmpeg

Do not add these dependencies to `worker/pyproject.toml`.

## 8. Implement RGB-D Gaussian training

Add a ScanLan Splatfacto configuration that:

* Initializes Gaussian centers and colors from `initialization.ply`.
* Uses Open3D’s metric camera poses.
* Renders RGB plus expected depth.
* Applies RGB L1, SSIM and masked robust depth losses.
* Excludes missing depth, sensor-range clipping and depth discontinuities.
* Gradually reduces depth-loss weight after geometry stabilizes.
* Saves resumable checkpoints.
* Exports `outputs/room-splat.ply`.
* Writes `outputs/splat-manifest.json` with source fingerprint, trainer versions, metric scale and coordinate convention.

gsplat directly supports combined RGB and expected-depth rasterization. [gsplat rasterization](https://docs.gsplat.studio/main/apis/rasterization.html)

Do not bake ScanLan’s non-uniform viewer transform into the PLY initially. Export:

```text
room-splat.ply
room-splat.transform.json
```

The Unity importer should apply the transform at the GameObject level. Correctly baking non-uniform transforms would require transforming each covariance and handling spherical-harmonic orientation.

## 9. Add photo and video sources

Extend the Tauri dialog/API and Svelte UI with:

* `Import photos…`
* `Import video…`
* Durable copying into `sources/<source-id>/originals`
* Media-source removal
* Source quality/status display

Processing:

1. Extract video frames through FFmpeg.
2. Apply blur and near-duplicate filtering.
3. Use COLMAP sequential matching for video.
4. Use exhaustive or vocabulary-tree matching for photos based on dataset size.
5. Report registered images, reprojection error and disconnected components.
6. Stop before training when registration quality is insufficient.
7. Convert COLMAP output into the canonical dataset.
8. Train standard Splatfacto using its sparse initialization.
9. Mark the result as arbitrary-scale until the user supplies a known distance.

COLMAP exposes appropriate reconstruction and matching modes for both ordered and unordered media. [COLMAP CLI](https://colmap.github.io/cli.html)

## 10. UI and export integration

Update:

* `src/App.svelte`
* `src/lib/api.ts`
* `src/lib/types.ts`
* `src/lib/components/PointCloudPreview.svelte`

Add:

* Source list: RGB-D phases, photo sets and videos.
* Artifact build checkboxes.
* Splat quality preset and iteration budget.
* Long-running job status and cancellation.
* Separate export buttons for cloud, mesh and splat.
* Artifact freshness/staleness indicators.
* Splat runtime diagnostics.

Default targets:

* RGB-D project: point cloud + mesh; splat optional.
* Photo/video project: splat.
* Mixed project: explicit source selection.

Ship splat export before embedded splat visualization. Later add a dedicated renderer and extend render mode to:

```ts
'points' | 'mesh' | 'splat'
```

The existing `THREE.Points` renderer must not be used to approximate Gaussian splats.

## PR sequence

1. Golden coordinate/export fixtures.
2. Additive schemas and migrations.
3. Durable artifact-job infrastructure.
4. Native full-resolution RGB capture.
5. Canonical posed-frame dataset.
6. Full-resolution mesh texturing.
7. Optional splat runtime and smoke test.
8. RGB-D depth-aware splat export.
9. Photo/video ingestion and RGB-only splats.
10. UI/export integration.
11. Optional embedded splat viewer.
12. Packaging, documentation and hardware validation.

Each PR should preserve existing tests and add focused coverage. Avoid mixing schema, capture, training and UI changes in one large PR.

## Final definition of done

* Old ScanLan projects still reconstruct and export.
* New captures retain native RGB for all supported sensors.
* Mesh textures visibly use native RGB detail.
* Kinect v2, Azure Kinect and Femto produce aligned metric cloud, mesh and splat artifacts.
* Photo folders and videos produce Gaussian splats with quality diagnostics.
* Jobs can be cancelled and resumed.
* The Gaussian PLY opens with correct scale, orientation and appearance in the target Unity renderer.
* CPU-only installations retain all current features and clearly report splat support as unavailable.
