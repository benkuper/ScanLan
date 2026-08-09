# Reconstruction pipeline

All representations share one quality-gated RGB-D trajectory. A visually attractive splat is not allowed to hide a broken metric path.

## Realtime pass

1. Validate and decode `SCANRGBD` frames.
2. Remove isolated depth returns without smoothing across supported edges.
3. Build persistent tensor or CPU RGB-D representations.
4. Estimate odometry, optionally initialized by the calibrated gyro delta.
5. Reproject metric depth and reject candidates with weak overlap, too few correspondences, high RMSE, or impossible motion.
6. On loss, test recent accepted anchors through the same gates.
7. Select keyframes by translation, rotation, and elapsed time.
8. Integrate selected frames into a bounded active sparse TSDF submap.
9. Move completed submaps to compact host geometry while retaining their rigid transforms.
10. Publish normal, coverage, and tracking-confidence point snapshots at independent adaptive rates, plus an optional 1 Hz mesh when headroom permits.

The low-resolution coverage field stores per-cell observation count, best projected pixel
density, pose confidence, and recent sequence. Green geometry has at least three independent
observations, yellow has two, orange is single-view, and purple is unknown. Guidance is derived
from measured coverage and tracking confidence, so the UI can request more parallax, a revisit,
or a return to trusted geometry without changing raw observations.

Kinect Fusion poses enter at step 4 but still pass ScanLan’s finite, rigid, physical-motion, overlap, and metric depth-residual checks.

Azure Kinect uses 2048×1536 RGB beside every 30 fps depth mode and 3840×2160 RGB beside the 15 fps wide/full mode; these are the highest synchronized combinations supported by the device. The archive rate remains independent, so high-resolution source RGB is retained only for selected production keyframes while tracking receives every aligned depth-rate frame.

## Production trajectory

For each take, the final pass prefers a complete validated live trajectory. If it is unavailable, it reruns RGB-D odometry from the archived frames. It then:

- selects quality-spaced keyframes;
- divides the trajectory into short overlapping fragments;
- registers adjacent fragments with colored/geometric refinement;
- retrieves a bounded set of nonlocal candidates;
- accepts loop closures only after strict fitness, overlap, RMSE, color, and correction gates;
- optimizes the fragment pose graph with uncertain loop edges;
- rejects the optimized graph if its correction exceeds global safety limits;
- interpolates rigid corrections smoothly over all camera poses.

Separate takes are registered through global features followed by quality-gated colored ICP. An ambiguous take fails visibly instead of being fused at an arbitrary transform.

## Optional LingBot depth refinement

Depth refinement is an offline production stage after the complete metric trajectory is accepted. Tracking, loop closure, take registration, and pose refinement always use original calibrated sensor depth. The isolated CUDA worker runs the pinned Apache-2.0 LingBot-Depth v0.5 code/model on `color/*.rgb` and `depth/*.u16`; both inputs are already on the depth-camera pixel grid, and the model output is required to have exactly that same width and height.

ScanLan never replaces any nonzero sensor measurement, including one outside the configured fusion range. A prediction may fill only an original zero-valued sensor hole and must pass all of these gates:

- the model validity mask and finite metric depth range;
- local depth-discontinuity rejection;
- agreement with reliable measured pixels in median, 90th-percentile, scale-bias, and inlier metrics;
- projection through the calibrated depth-to-native-RGB transform into the true RGB field of view;
- reprojection agreement from independent camera centers in up to four nearby posed keyframes, with two confirmations when available.

Validated caches are fingerprinted by source depth/color files, calibration, poses, implementation version, code revision, model revision, and model SHA-256. Measured and refined rasters are stored separately with generated-pixel masks and 8-bit confidence. Final TSDF/surfel fusion integrates measured depth twice and the measured-plus-generated raster once, giving generated geometry half weight. Canonical 2DGS datasets retain both confidence and provenance masks; generated depth has lower robust-loss weight and lower seed opacity, while measured seeds win voxel conflicts.

## Point cloud

Normal room-scale settings use weighted TSDF fusion. Fine spacing uses a memory-bounded voxel/surfel path so the GPU does not allocate a room-sized dense volume. The result is filtered, downsampled at the requested metric spacing, colored, and saved as binary PLY.

## Triangle mesh

Before texturing, the optional CGAL repair pass analyzes topology and projects candidate hole samples into the original RGB-D frames. It fills measured dropouts only when multi-view depth supports the proposed surface and preserves free-space openings, occluded cavities, oversized scan boundaries, and unknown holes. Faithful, Architectural, and Natural profiles control only authorized patch geometry; an optional watertight PLY remains a separate derivative. See [mesh-repair.md](mesh-repair.md).

The mesh is extracted from the same final CUDA/CPU TSDF used by the point output, so a combined point+mesh build does not fuse the room twice. Geometry uses every accepted depth keyframe; texturing retains up to 24 views by greedily maximizing surface coverage and projected source-pixel density. The surface is cleaned, simplified to a bounded triangle budget, welded, and indexed. Azure/Femto texturing uses calibrated native RGB whenever its bounded encoder produced the frame; a missing native image switches that frame to depth-aligned RGB with depth-camera calibration. Kinect always uses the SDK’s exact depth-aligned color rather than a hard-coded approximation of unavailable reusable color calibration:

- project geometry through depth and RGB calibration;
- conservatively refine RGB-D poses geometrically against the fused surface, then run a bounded rigid color-map optimization; corrections outside strict millimetre/degree gates are discarded;
- enforce captured-depth visibility and use a mesh-vertex z-buffer for localized depthless photos;
- estimate per-channel gains and biases from overlapping observations in linear RGB;
- fit a smooth low-frequency correction field per image, leveling local exposure, white balance, and vignetting without averaging away detail;
- optimize camera labels over adjacent coplanar faces, forming coherent patches instead of triangle-scale camera switches;
- prefer frontal, close, high-pixel-density observations and keep one sharp source per patch;
- pack every corrected source image once into a padded shared atlas page, allowing same-camera faces to reuse native detail instead of receiving independent microcharts;
- retain blended multi-view vertex color only as a fallback for surfaces no camera observes completely.

Outputs are `room-mesh.obj`, `room-mesh.mtl`, and `room-texture.png`.

### Supplemental high-resolution photos

After an initial mesh build, the Reconstruct workspace can import overlapping JPEG, PNG, TIFF, or WebP scene photos. The localization worker detects SIFT features in the new photo and selected depth-aligned RGB-D references, lifts reference features into metric world coordinates with captured depth, searches plausible focal lengths, and estimates the camera with PnP-RANSAC. Inlier-count, inlier-ratio, and reprojection-RMSE gates reject ambiguous poses.

Selected files are registered immediately in `supplemental-photos.json`, and atomic progress is checkpointed in `outputs/photo-localization-progress.json`. The Reconstruct workspace restores that state after reload and lists queued, accepted, and rejected photos. Accepted entries include a 0–100 matching-quality score plus inlier and reprojection diagnostics; any entry can be removed without deleting its original source file.

Accepted images are orientation-normalized and copied losslessly into `supplemental/`; calibrated pinhole cameras are stored in the manifest's `photos` collection. The textured mesh is marked stale, and its next rebuild includes accepted photos in coverage selection, visibility testing, radiometric calibration, coherent labeling, and atlas packing. The standalone equivalent is:

```powershell
scanlan-worker.exe localize-photos C:\path\to\project C:\photos\view-01.jpg C:\photos\view-02.jpg
```

Localization requires the OpenCV feature runtime included in packaged workers. Failed photos remain outside the texture-baking `photos` collection but persist in the `attempts` registry with their match or pose-validation reason.

## 2D Gaussian surface

The splat target uses tangent-aligned 2D Gaussian discs rather than unconstrained volumetric blobs. Initial seeds come from the same metric keyframes and include local normals, anisotropic pixel footprints, depth masks, and measured/generated confidence. Native Azure/Femto lens distortion is removed from RGB, reprojected depth, masks, and provenance before optimization so every training ray matches the pinhole 2DGS rasterizer. Training combines photometric, SSIM, confidence-weighted robust expected-depth, normal, and Gaussian-distortion terms with bounded camera-pose refinement.

For a 12 GB GPU, the canonical builder retains up to 600 views using camera-position, direction, roll, time, and take-boundary coverage, then projects metric depth and undistorts RGB directly onto a 960 px pinhole grid. Compact integer images remain in pinned host memory behind a four-frame LRU; shuffled views are scheduled in cache-local blocks and only the active view is transferred to CUDA. The trainer enforces a two-million-Gaussian hard ceiling at this VRAM tier. Densification stops before a single growth cycle could cross that ceiling, and checkpoints are exported atomically.

The exported PLY is interoperable with 3DGS tooling by flattening the third scale axis. `room-splat.transform.json` records display conversion separately. This is the production surface-splat path; the conventional mesh remains the production triangle representation.

## Photo/video 3D Gaussian splat

Ordinary media uses a separate photoreal path because it has no metric depth surface to constrain 2D discs. Photos are orientation-normalized without upscaling. Video is decoded with the bundled FFmpeg/PyAV runtime, evaluated at three times the target output rate, reduced to the sharpest frame in each time bucket, pruned for near-duplicates, and capped at 600 selected frames by default.

PyCOLMAP extracts bounded high-density SIFT features, uses guided geometric verification, performs exhaustive matching for normal photo sets and quadratic sequential matching for long videos, and incrementally reconstructs multiple candidate models. All frames from one video share one physical camera throughout feature extraction and bundle adjustment; ScanLan rejects a video if canonical intrinsics drift after undistortion. A solve must register at least 45% of the usable input views and produce at least 100 reliable sparse tracks. While mapping runs, registered-camera, best-model, model-attempt, and elapsed-time telemetry is published once per second. The largest consistent model is bundle-adjusted, undistorted at source resolution, and converted into canonical schema-3 pinhole cameras. Registration ratio, excluded views, model count, reprojection error, track length, and warnings remain in `dataset.json`.

Reliable COLMAP points initialize anisotropic 3D Gaussians with local-spacing-derived scales. Training uses packed gsplat rasterization, L1+SSIM, degree-three spherical harmonics, and bounded camera-pose refinement. Photo sets can use a regularized per-view RGB log-gain/bias model, with the first/median-exposure anchor fixing its gauge. A single locked-settings video keeps appearance fixed so color correction cannot hide geometric disagreement. Checkpoints, live previews, the final canonical PLY, refined cameras, and sidecars use the same atomic publication policy as RGB-D 2DGS.

### Rebuild and cache policy

Decoded, sharpness-selected media observations are immutable and cached separately from camera analysis and output datasets. The Reconstruct workspace exposes the restart boundary for every new build:

- **Cached analysis** reuses compatible decoded views, camera solutions/localizations, RGB-D poses, and geometry caches while rebuilding the selected outputs.
- **Camera analysis** keeps decoded and selected media views, but discards camera solutions/localizations and every downstream training dataset.
- **Media decode** discards prepared media views as well, forcing video decode, sharp-frame selection, and all downstream analysis.
- **Re-run RGB-D tracking and fusion** independently discards cached RGB-D poses and geometry while retaining decoded media observations.

Source and setting fingerprints remain authoritative: choosing reuse never forces an incompatible cache hit. Starting a new build also clears the previous Gaussian training checkpoint; only the explicit interrupted-job **Resume checkpoint** action continues one.

## Failure policy

- Queue pressure drops stale work instead of increasing latency.
- Invalid tracking freezes fusion and increments rejection counters.
- A broken journal falls back to offline odometry.
- A failed loop-graph optimization falls back to the previously validated trajectory.
- CUDA operation failure can fall back to the CPU Open3D path for point/mesh reconstruction.
- Gaussian training requires CUDA and reports a missing runtime instead of silently changing algorithms; media camera solving remains CPU-capable.
- Raw schema-3 capture data is never deleted by reconstruction cancellation.

## Benchmark gate

Before changing thresholds, record a fixed suite with a short closed loop, blank walls, thin objects, reflective surfaces, rapid rotation, and one deliberate relocalization. Track:

- accepted/rejected frames and relocalization time;
- trajectory ATE/RPE when reference motion exists;
- depth reprojection RMSE and overlap;
- mesh accuracy, completeness, duplicate surfaces, and holes;
- held-out RGB PSNR/SSIM and metric depth error for 2DGS;
- tracking fps, extraction latency, queue drops, peak VRAM, and wall time.

Production acceptance is metric first: no representation passes if its trajectory or depth residual fails, regardless of appearance quality.
