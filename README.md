# ScanLan

ScanLan is a Windows-first, realtime RGB-D reconstruction application for Kinect v2, Azure Kinect DK, and Orbbec Femto Mega. A capture can become three production outputs from one calibrated trajectory:

- a metric colored point cloud (`PLY`);
- a textured triangle mesh (`OBJ` + `MTL` + `PNG`);
- a metric depth-aware 2D Gaussian surface (`3DGS-compatible PLY`).
- a photoreal anisotropic 3D Gaussian splat reconstructed from ordinary photos or video (`PLY` + compact live preview).

The application supports two quality-gated source paths: **Capture RGB-D → Reconstruct** and **Import photos/video → Solve cameras → Reconstruct**. Both finish in the same inspect/export workspace and publish interoperable 3DGS PLY files. Project and capture manifests use schema 3.

## Design

```mermaid
flowchart TD
    Camera["RGB-D camera"] --> Capture["Native capture worker"]
    Media["Photos or video"] --> Proposal["DA3 / MapAnything camera proposal"]
    Proposal --> SfM["Verified guided pairs · COLMAP BA · undistortion"]
    Capture -->|"full sensor rate · SCANRGBD v1"| Tracker["Realtime tracker"]
    Capture -->|"bounded archive queue"| Archive["Schema 3 capture"]
    Tracker --> Live["In-memory points / mesh"]
    Tracker --> Journal["Quality + pose journal"]
    Archive --> Final["Production reconstruction"]
    Journal --> Final
    Final --> Outputs["Points · mesh · 2DGS"]
    SfM --> PhotoGS["Exposure-compensated photoreal 3DGS"]
    PhotoGS --> Outputs
```

The sensor thread never waits for disk, JPEG compression, the UI, or TSDF extraction. Its stream and archive writers are bounded and latest-wins. If downstream work falls behind, stale frames are dropped, sequence gaps are counted, and latency stays bounded.

In the Capture workspace, a connected Azure Kinect or Femto Mega starts this pipeline in non-recording preview mode. The camera connection and realtime engine remain warm, so **Start capture** only opens the archive gate; preview RGB-D and IMU samples are never written into the take. Preview does not run odometry or retain tracking anchors: a fresh tracker takes the first usable recorded depth frame as its identity pose immediately after the archive gate opens. The viewport deliberately uses separate streams: idle Capture shows the current calibrated camera point cloud, recording switches to the accumulated quality-gated TSDF reconstruction, and stopping a take resumes the camera view. The final fused live point cloud is preserved with the take and is shown in Reconstruct even before production artifacts are built. Geometry polling is independent from status polling, so viewport responsiveness does not change capture resolution, archived pixels, voxel size, odometry thresholds, or drift gates. Preview stops before production reconstruction so it does not compete for the camera or CUDA device.

Realtime processing uses three independent stages:

1. decode and edge-aware depth-speckle rejection;
2. persistent RGB-D odometry with an optional calibrated gyro prior, deliberate handheld-motion limits, metric overlap/RMSE gates, and fail-closed recovery that must remain within 15 cm / 10 degrees of the last trusted pose and agree for three consecutive frames before integration;
3. bounded sparse-TSDF submaps with compact host caching, adaptive point/coverage publication,
   tracking-confidence overlays, and an optional 1 Hz mesh that pauses under pressure.

The live map has a hard memory ceiling. Travel, rotation, voxel pressure, keyframe count, or a
tracking discontinuity closes the active submap; completed submaps remain visible as immutable
host geometry and can later move through pose-graph corrections. The capture viewport can
switch between normal color, coverage, tracking, and confidence views without changing the
sensor, archive, tracking, or fusion rates.

Nonlocal live loops are queried only at bounded submap boundaries and accepted only after
strict metric ICP verification. Accepted pose-graph corrections move existing rigid submaps
over 350 ms without reintegration or duplicate geometry, while `live_loops.jsonl` records every
decision for independent production revalidation.

The compact `tracking.jsonl` journal feeds accepted live poses into the production pass. A rejected live pose does not discard its archived pixels: the final pass keeps consecutive RGB-D frames for fresh offline odometry, then refines short trajectory fragments, verifies nonlocal loop candidates, globally optimizes a pose graph, registers separate takes, and rebuilds the selected outputs from archived calibrated RGB-D data.

See [architecture](docs/architecture.md), [archive format](docs/scan-format.md), and [reconstruction details](docs/reconstruction.md).

## Camera support

| Camera | Connection | Tracking pose/input | Notes |
|---|---|---|---|
| Kinect v2 | USB 3 | Kinect Fusion pose, validated again by ScanLan | Kinect for Windows SDK 2.0 |
| Azure Kinect DK | USB 3 | RGB-D odometry + calibrated gyro prior | Azure Kinect Sensor SDK 1.4.x |
| Orbbec Femto Mega | USB or Ethernet | RGB-D odometry + calibrated gyro prior | Orbbec SDK v2; network default port 8090 |

Azure Kinect and Femto Mega expose these depth profiles:

| FOV | Sampling | Depth size | Sensor rate |
|---|---|---:|---:|
| Narrow | Full | 640×576 | 30 fps |
| Narrow | 2×2 binned | 320×288 | 30 fps |
| Wide | Full | 1024×1024 | 15 fps |
| Wide | 2×2 binned | 512×512 | 30 fps |

Azure Kinect pairs 30 fps depth with its highest compatible 2048×1536 RGB mode. The 15 fps wide/full profile records 3840×2160 RGB. This avoids the unsupported 30 fps + 4K configuration while preserving maximum texture detail at each sensor rate.

The archive rate is independent of the sensor/tracking rate. For example, a 10 fps archive still tracks a 30 fps camera stream.

The Capture workspace exposes the modern cameras' sensor-side RGB controls: compatible stream resolution, automatic or locked exposure/gain, automatic or locked white balance, brightness, contrast, saturation, sharpness, backlight compensation where supported, and 50/60 Hz anti-flicker. Native-RGB archive size and JPEG quality are separate controls; for maximum texture detail use native size and quality 95-100. Kinect v2 remains fixed at 1920x1080 color because Kinect for Windows SDK 2.0 exposes its color-camera settings as read-only values.

Femto Mega also exposes exact accelerometer and gyroscope rate/range profiles. A faster rate improves temporal sampling but costs bandwidth and CPU. A narrower full-scale range provides finer quantization; a wider range such as +/-8 g prevents clipping but does not increase precision. Azure Kinect's SDK provides a factory-calibrated IMU stream without configurable rate or range, and Kinect v2 has no accessible IMU.

## Windows setup

Install:

- Visual Studio 2022 with **Desktop development with C++** and CMake;
- Rust stable with the MSVC target;
- Node.js 20 or newer;
- Python 3.10–3.12 for the reconstruction worker;
- the SDK for at least one supported camera.

Then:

```powershell
npm install
npm run prepare:runtime
npm run debug
```

`prepare:runtime` builds both native capture executables. A camera SDK that is not installed produces a small unavailable-backend stub; it does not prevent other cameras from building. The command also packages the Open3D reconstruction worker.

For an optimized local build:

```powershell
npm run release
```

For the regular NSIS installer:

```powershell
npm run tauri -- build
```

## RTX 5080 / 12 GB setup

The stock Windows Open3D wheel is CPU-only. For Blackwell CUDA tracking and fusion, install CUDA Toolkit 12.8+ (CUDA 13.x is recommended), then build the project-local Open3D 0.19 wheel:

```powershell
npm run prepare:runtime
npm run prepare:cuda
```

The build targets compute capability 12.0 and repackages the extracted worker at `worker\dist\scanlan-worker\scanlan-worker.exe`. Keeping its Open3D/CUDA runtime extracted avoids unpacking almost 1 GB every time preview or reconstruction starts. Set `SCANLAN_DEVICE=cpu` before launching to diagnose the CPU path.

Gaussian reconstruction and optional learned RGB-D refinement use two supervised CUDA runtimes. The splat runtime owns PyCOLMAP, PyAV, gsplat, and training; the separate geometry runtime owns LingBot-Map, LingBot-Depth, MapAnything, and DA3 Nested Giant-Large 1.1 so model memory is released with each inference process. Both require Python 3.11 and are built together:

```powershell
npm run prepare:splat
npm run package:splat
```

The result is `build/ScanLan-splat-portable.zip`, containing `splat-runtime/` and `geometry-runtime/`. The build downloads pinned model assets once, verifies their digests, and bundles them for offline use. RGB-D projects can select **LingBot-Depth**, **MapAnything Apache**, or **DA3 Max** under Depth refinement. DA3 Max uses the strongest refreshed Nested Giant-Large checkpoint and direct Gaussian head; its checkpoint and derived output are restricted to noncommercial use under CC BY-NC 4.0. All paths keep bounded memory, publish atomic checkpoints, and stream a compact preview during training.

Video-only projects can optionally enable the disabled-by-default **Progressive learned-depth preview**. It publishes bounded local LingBot submaps during ordered inference, colors geometry by confidence, and always labels scale as model-metric unverified. The provisional map is display-only and cannot bypass production camera or alignment gates.

Camera, scale, depth, free-space, and point acceptance are defined once in the shared
[`scanlan-validation` engine](docs/validation-engine.md). The RGB-only preview and production
RGB-D worker emit the same versioned reports, and frozen runtimes package the validator for
offline use. Learned backends may propose geometry but cannot promote unverified scale or bypass
measured-depth/free-space evidence.

LingBot-Depth consumes the archived RGB8 image that is already aligned to the depth grid and returns the same raster dimensions, so no post-hoc RGB warp is inferred. ScanLan runs it only after metric camera poses have been recovered. Every valid sensor depth remains unchanged; predicted pixels are accepted only in sensor holes after model-mask, depth-edge, metric-scale, calibrated native-RGB field-of-view, independent-viewpoint, and multi-view reprojection gates. Accepted pixels carry explicit provenance and lower fusion/training confidence. If a frame fails the metric gate, its raw calibrated depth is used unchanged.

MapAnything uses the same immutable aligned RGB-D archive but predicts in its processed image grid. ScanLan reverses the cover-resize/center-crop transform, calibrates the smooth model residual from sensor anchors, and validates only on independent held-out anchors. Unsupported large holes and any proposal that fails metric, RGB-coverage, multi-view, or free-space evidence remain sensor-only. For photos and videos of at most 32 selected views, MapAnything also proposes cameras and dense depth as a challenger; COLMAP agreement selects or rejects it, and image-only scale remains `MODEL_METRIC_UNVERIFIED` until anchored.

DA3 Max uses DA3NESTED-GIANT-LARGE-1.1 for any-view cameras, metric-aware pose-conditioned depth,
and direct Gaussian proposals. Long media runs in bounded 24-frame windows with six-frame overlap;
every join must pass camera-center and rotation continuity gates before the complete proposal is
compared with COLMAP and the other learned backends. The direct Gaussian head runs inside those
same bounded windows. If its measured CUDA allocation exceeds available headroom, the isolated
worker records that failure and retries with the model's confidence-gated camera/depth output;
source-resolution gsplat optimization then remains the final quality stage. A versioned manifest
distinguishes sparse SfM, dense surface, and direct learned initialization. The sidecar preserves the
direct head's learned opacity independently from geometric confidence and retains anisotropic scale
axes; opacity-free point previews are not used as a quality judgment for this representation.

Recommended starting profile on the specified laptop:

- narrow/full depth at 30 fps for normal indoor scans;
- 10–15 fps archive rate;
- 8–12 mm fusion voxels for rooms, 5–8 mm for smaller objects;
- live points while maximizing tracking headroom, or the 1 Hz live mesh when desired;
- 30,000 Gaussian iterations for a normal production build; 45,000-60,000 can improve a well-covered high-detail photo/video capture.

## Photos and video

Choose **Import photos or video for Gaussian splatting…** in Capture. Imported sources are copied into the project so the job is durable. The production pass then:

1. orientation-normalizes photos and selects the sharpest non-duplicate video frame in each time bucket;
2. asks DA3 (or bounded MapAnything fallback) for a camera proposal and uses it to select a connected, bounded pair graph;
3. extracts source-detail ALIKED/LightGlue features (SIFT fallback), geometrically verifies every proposed pair, and recovers missing cameras through nearby verified learned views;
4. expands to conventional matching when the guided solve misses its quality gate, then robustly bundle-adjusts the strongest model and undistorts its registered images to canonical pinhole cameras;
5. publishes learned-scale colored points and a confidence-gated triangle surface from the accepted dense prior when selected;
6. initializes Gaussians through the explicit sparse/dense/direct contract, trains bounded global L1+SSIM appearance with degree-three spherical harmonics, bounded camera refinement, and per-view RGB exposure compensation, then covers every calibrated source-resolution tile before publication.

Media splats must also pass a deterministic five-view raw-render gate (median PSNR, SSIM, and L1).
An undertrained or divergent candidate is not published; its final atomic checkpoint is retained so
the job can resume with a longer optimization budget.

Material-aware production starts from a separate fail-closed foundation. ScanLan converts embedded
ICC input to canonical sRGB, decodes the exact IEC transfer into content-addressed linear-light
frames, and keeps material identity separate from overlapping glass, mirror, specular, emissive,
thin-geometry, dynamic, and sky risks. Frozen commercial/research candidate manifests prevent
noncommercial or unverified assets from entering a commercial pack. Material Anything, RGB-to-X,
and DiffusionRenderer remain bake-off candidates until they pass real-capture quality, multiview,
calibration, and 12 GB memory gates; see [the P13 foundation](docs/material-radiometric-foundation.md).

The P14 two-pass engine keeps that gate intact while making material inference operationally useful.
It samples the measured camera path for a bounded coarse optical-risk pass, then chooses final views
by incremental 3D surface coverage with extra authority for glass, mirror, specular, emissive,
thin, dynamic, and sky warnings. Calibrated final predictions are visibility-, pose-, angle-, and
confidence-weighted onto the production surface. Material identity is fused as multiview evidence,
while a strong optical warning from one sound view survives averaging. The versioned surface
sidecar records support, effective view count, confidence, and connected material/risk regions; see
[the P14 analysis](docs/two-pass-material-analysis.md).

P15 makes those risks actionable without letting labels hallucinate geometry. Material evidence can
only reduce measured/generated/learned depth confidence; glass, mirror, thin, dynamic, and sky
regions conservatively veto unsupported repair. A second-pass proposal must pass provenance-specific
confidence, independent-view, held-out metric residual, displacement, and triangle-topology gates.
Missing material output is a neutral no-op, and protected surfaces move only through a stricter
multiview recovery gate; see [the P15 geometry policy](docs/material-aware-geometry.md).

The dense initialization sidecar is also the shared point/mesh fusion contract. It retains source
ownership, confidence, provenance, orientation, and footprint. Media-only point and mesh artifacts
are correctly labeled non-metric. In a hybrid project, learned media geometry is robustly aligned
through independently localized cameras and can only fill space outside calibrated RGB-D support;
failed camera agreement excludes it without degrading the metric result.

For the selected mesh, **Neural SDF refinement (Max Quality)** optionally fits a continuous
signed-distance surface in the isolated CUDA runtime after camera/depth validation. It is strictly
fail-closed: deterministic held-out SDF error, bounded displacement, triangle orientation,
degeneracy, and an independent reconstruction-worker check must all pass before the candidate can
continue to repair and multiview texturing. Otherwise the validated TSDF or learned dense mesh is
kept unchanged. The exact decision is saved in `outputs/neural-sdf-report.json`.

The dataset manifest records proposal backend, guided/recovery/fallback pair counts, geometric
verification evidence, recovered cameras, registration ratio, excluded views, model count,
reprojection error, track length, and warnings. Disconnected views are reported rather than forced
into the splat. Video keyframes are selected adaptively from optical flow at a 15 fps analysis rate;
3,000 retained frames is a crash-safety ceiling, not a sampling target.

For strong results, keep 60-80% overlap, translate as well as rotate the camera, lock focus/exposure when possible, avoid motion blur, and revisit the start of the path. A sparse panorama captured from one fixed point does not contain enough parallax for a full scene reconstruction.

The standalone preparation path is:

```powershell
scanlan-splat.exe prepare-media --project C:\path\to\project --source C:\photos\view-01.jpg --source C:\capture.mp4
scanlan-worker.exe fuse-dataset C:\path\to\project C:\path\to\project\outputs\cache\datasets\media-current.json --targets point_cloud,textured_mesh
scanlan-splat.exe train --project C:\path\to\project --dataset C:\path\to\project\outputs\cache\datasets\current.json --iterations 30000
```

## Capture practice

- Move steadily and retain roughly 40–70% of the previous view.
- When tracking searches, return to the last well-textured surface instead of continuing into unknown space.
- Revisit the start before stopping; this gives the production pass a strong loop-closure opportunity.
- Use corners, furniture, rocks, or temporary texture/marker boards around blank walls.
- Keep reflective glass and direct sunlight out of the depth camera when possible.
- Start a new take while viewing geometry already captured in the previous take.

For additional texture detail, build the mesh once, then use **Add and localize photos…** in Reconstruct. Use overlapping, sharp scene photos with fixed focus and white balance where possible. Accepted photos are metric PnP-localized from depth-backed RGB-D features and join the next mesh rebuild; ambiguous poses are reported and excluded. The worker command is `scanlan-worker localize-photos <project> <photo> [<photo> ...]`.

## Deterministic replay

The reconstruction worker exposes the same versioned RGB-D protocol used by physical cameras:

```powershell
worker\dist\scanlan-worker\scanlan-worker.exe replay C:\Scans\Room\phases\<phase-id>
worker\dist\scanlan-worker\scanlan-worker.exe realtime --session C:\Temp\scanlan-replay --mode mesh --voxel-size 0.01
```

`replay` emits archived raw RGB-D frames, including frames rejected by the original live run, so tracker changes can be evaluated from identical input. Unit tests cover binary framing, truncation, queue overload, pose round-trips, quality gates, relocalization, and engine geometry messages.

## Verification

```powershell
npm run check
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
.\build\worker-venv\Scripts\python.exe -m unittest discover -s worker\tests -v
.\splat-worker\.venv\Scripts\python.exe -m unittest discover -s splat-worker\tests -v
.\splat-worker\.venv\Scripts\python.exe -m unittest discover -s material\tests -v
```

Native capture workers must also be compiled on Windows against their vendor SDKs; portable header tests cannot validate those SDK calls.

## Repository layout

- `src/` — focused Svelte UI and GPU viewer
- `src-tauri/` — project state, process supervision, artifact jobs, and exports
- `native/common/` — versioned stream, bounded archive, JPEG, and gyro utilities
- `native/kinect-capture/` — Kinect v2 + Kinect Fusion capture
- `native/modern-capture/` — Azure Kinect and Femto Mega capture
- `worker/` — realtime and final Open3D/NumPy reconstruction
- `splat-worker/` — isolated COLMAP/PyAV media solver and CUDA 2DGS/3DGS trainer
- `geometry-worker/` — isolated LingBot-Map, LingBot-Depth, MapAnything, and DA3 model process
- `scripts/` — Windows build and packaging entry points

Licensed under [GPL-3.0-only](LICENSE).
