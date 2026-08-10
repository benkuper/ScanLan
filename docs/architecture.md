# Runtime architecture

ScanLan separates realtime latency from archival throughput and final quality. Every boundary is explicit and versioned.

## Processes

| Process | Owns | Must not own |
|---|---|---|
| Tauri application | lifecycle, project state, status snapshots, exports | camera SDK state or reconstruction kernels |
| Native capture worker | camera SDK, calibration, synchronized RGB-D, IMU conversion | UI rendering or Open3D |
| Reconstruction worker | tracking, relocalization, TSDF, pose graph, point/mesh build | camera SDK |
| Splat worker | Photo/video decoding, COLMAP camera solving, CUDA 2DGS/3DGS optimization, checkpoints | Learned-model loading, RGB-D capture, tracking, pose recovery, and TSDF quality decisions |
| Geometry worker | Pinned LingBot-Map, LingBot-Depth, MapAnything, and DA3 Nested Giant-Large CUDA inference, model lifecycle, lossless array publication | Camera solving, production validation, fusion, or Gaussian optimization |

Both reconstruction runtimes import one versioned, NumPy-only validation engine. It owns generic
SE(3) camera continuity, robust Sim(3) scale evidence, metric-depth agreement, ray free-space
consistency, and point admissibility. Learned adapters propose geometry; they do not define their
own acceptance policy. Reports remain serializable across process boundaries. See
[validation-engine.md](validation-engine.md).

The splat worker also owns the opt-in Max Quality neural-SDF optimizer so its PyTorch/CUDA
allocation cannot leak into the lightweight reconstruction runtime. The reconstruction worker
serializes an immutable indexed candidate surface, invokes the isolated process, and validates the
returned displacement independently before repair or texturing. A worker crash, CUDA error,
rejected held-out fit, or malformed output therefore cannot replace the baseline mesh.

Gaussian datasets publish a schema-1 initialization contract identifying sparse SfM points, dense
surface samples, or direct learned anisotropic Gaussians. The contract selects 2D/3D representation,
parameter sidecar, and densification policy; it also prevents direct-model opacity from being reused
as geometric confidence. Training first uses bounded global rasters, then schedules every calibrated
source-image tile at native pixel density with a crop-adjusted pinhole principal point. This keeps
CUDA raster memory bounded while making source-resolution coverage a publication gate. See
[gaussian-production-benchmark.md](gaussian-production-benchmark.md).
Media splats then render five deterministic calibrated views without the training-only exposure
transform. Median PSNR/SSIM/L1 gates the interoperable PLY itself; rejection preserves a resumable
final checkpoint and cannot publish the candidate as ready.

The realtime engine is started and reports `ready` before the camera is opened. Tauri pipes camera stdout directly into engine stdin and drains engine stdout on a dedicated thread. Sensor stderr goes to `sensor.log`, so a full pipe cannot stall capture.

## Live data path

`SCANRGBD` version 1 frames carry:

- sequence and device timestamps;
- pinhole depth-grid dimensions and intrinsics (Femto Mega native depth is calibrated and rectified before transport);
- depth scale and valid range;
- unsigned 16-bit depth and aligned RGB8;
- optional gyro delta quaternion;
- optional calibrated camera pose;
- the Kinect X-mirror flag.

The packed header is 164 bytes. Readers reject unknown versions, impossible calibration, oversized images, and payload-length mismatches before allocating geometry.

The capture-side stream queue has capacity 3 and discards its oldest unpublished item on overload. The Python reader feeds a latest-frame queue of 4; accepted keyframes feed a mapping queue of 8. None of these queues can grow without bound.

## Engine data path

`SCANENG1` version 1 multiplexes three message kinds:

- JSON status;
- packed point snapshots (`K2P1`, maximum 150,000 preview points);
- packed indexed meshes (`K2M2`, maximum 150,000 preview triangles).

Reconstruction 2.0 contract messages extend that same framed transport without breaking the
binary geometry readers: kind 4 is the raw-camera point preview, kind 5 is a versioned coverage
summary, and kind 6 is the ordered live-submap descriptor set. Status messages carry explicit
tracking state/confidence, pose uncertainty, pose/map latency, map-update rate, bounded-memory
counters, submap residency, queue pressure, degradation level, scale status, and whether
integration is frozen. A `ready` status also declares the capture/preview failure policy.

Tauri validates message sizes and stores only the newest point and mesh packets in memory. UI polling returns a packet only when its frame sequence is newer than the caller’s. Reconstruction geometry is never polled from a growing file.

## Archive path

Archival depth/aligned-color writes and native-RGB JPEG compression happen behind bounded queues. The camera loop moves frame buffers into the queue and immediately returns to acquisition. If storage cannot keep up, the oldest pending archive frame is discarded and the drop is persisted in phase metadata.

Modern-camera RGB controls are applied before video streaming starts. Exposure is represented in microseconds across backends (Femto Mega converts to its 100 microsecond property units), while white balance is Kelvin. Orbbec integer properties are clamped and snapped to the range/step reported by the connected firmware and adjustments are logged. Manual Femto IMU requests select an exact advertised stream profile; a missing profile is a startup error rather than a silent fallback. The requested RGB and IMU configuration is persisted in `phase.json` for capture provenance.

`live.json` is a tiny, atomically replaced sensor heartbeat. During capture the UI reads its monotonic `frameCount`; it does not rescan `frames.csv`. The complete CSV is counted only during recovery or abnormal termination.

## Tracking state machine

The tracker persists across frames. A finite transform alone is insufficient for acceptance. A candidate must pass:

- metric depth correspondence count;
- overlap and inlier-ratio thresholds;
- depth RMSE;
- translation and angular-velocity limits.

On failure, the map freezes and the tracker searches a rotating bank spanning the complete accepted capture. Local continuation remains subject to strict continuity and IMU limits; a strong saved-keyframe match may instead relocalize anywhere in the known map after three consistent observations. The lock frame is not fused, and mapping resumes only on the next independently validated frame. Tracking acceptance and irreversible fusion use separate gates: a marginal pose may preserve local tracking continuity, but only high-overlap, high-inlier, low-residual keyframes reach the mapping thread.

Every decision is appended asynchronously to `tracking.jsonl`; no image data is duplicated. Raw RGB-D archival is intentionally independent from live pose acceptance: a rejected frame remains recoverable evidence, but it never enters the live TSDF map. The UI therefore reports raw archived, tracked, rejected, and fused-keyframe counts separately. The final loader matches entries by sensor sequence, excludes explicit rejections from the live pose seed, and can recover archived frames during offline optimization. The archive replay command deliberately includes rejected frames and omits derived journal poses so tracker changes remain testable.

## Shutdown

Normal stop creates `stop.flag`, lets the capture worker flush its archive, drains the mapper for a final geometry snapshot, closes the engine, and then publishes the completed phase manifest. Timeouts are bounded; a stuck child is terminated rather than leaving the UI indefinitely busy.

Unexpected phases are recovered from their manifest and CSV at the next launch. Derived jobs have independent checkpoints and can be cancelled without deleting raw captures.

Optional LingBot-Depth, MapAnything, or pose-conditioned DA3 inference is launched as a child of the reconstruction job only after pose recovery. It reads an immutable JSON request and aligned raw frame paths, writes atomic NumPy predictions, observes the shared cancellation flag between frames, and never mutates an archive. The reconstruction worker—not the model process—owns held-out metric, RGB-coverage, multi-view, free-space, and provenance validation before publishing an immutable fingerprinted cache.

LingBot-Map, LingBot-Depth, MapAnything, and DA3 execute only in the dedicated geometry worker. Media
preparation writes a schema-1 request containing ordered absolute image paths, calibrated rays,
selected output indices, a hard seed limit, and the shared cancellation path. The geometry
worker owns the model and CUDA cache for exactly one request, then atomically publishes typed
NumPy arrays plus small JSON metadata. The caller validates shape, finiteness, camera
transforms, intrinsics, scales, quaternions, ownership, and confidence before using the result.
No Python object, CUDA tensor, or model state crosses the process boundary. See
[geometry-worker.md](geometry-worker.md).

All production geometry backends converge on the versioned `dense-surface-samples-v1` contract:
points, colors, oriented surface footprints, confidence, provenance, and source-frame ownership.
RGB-D sensor evidence is authoritative in occupied metric voxels; validated generated and learned
evidence fills missing support. Media-only artifacts retain learned scale. Hybrid learned geometry
must first pass a robust Sim(3) camera-agreement gate against independently localized media views,
and is rejected without affecting the calibrated reconstruction when that gate fails.

For ordinary photos and video, DA3 (or bounded MapAnything fallback) now runs before production
SfM. The splat worker accepts only a finite full-length proposal, then converts learned centers,
view directions, confidence, and temporal order into an explicit bounded pair list. COLMAP's
[imported pair matcher](https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.match_image_pairs)
still owns descriptor matching and two-view geometry. A targeted second pair list attempts to
recover missing learned views; a weak result expands to conventional matching. The best geometric
model receives a final robust global
[bundle adjustment](https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.bundle_adjustment)
before alignment gates decide whether learned dense geometry may seed optimization.

When the experimental RGB video preview flag is enabled, the same request also publishes bounded
eight-frame learned-depth submaps during causal inference. The splat worker converts the latest
validated snapshot to `build-preview.json`; the desktop displays confidence, drift risk, submap
count, rejection count, and the mandatory `MODEL_METRIC_UNVERIFIED` scale label. This preview is
never read back into the production solve. See [rgb-video-preview.md](rgb-video-preview.md).

Every active capture owns its sensor process, stdout relay, and realtime engine as one supervised unit. Dropping that unit after any startup, storage, or state error terminates all three, so a failed command cannot leave the camera or GPU worker running in the background.

Project and preference manifests are serialized to uniquely named sibling files, flushed to storage, and atomically replaced. Concurrent status or settings updates therefore cannot collide on one shared temporary path, and a failed publication leaves the previous valid manifest intact.

## Bounded live submaps

The Reconstruction 2.0 mapper owns exactly one active sparse voxel-block grid. Its capacity is
derived from the configured live-map budget (1,024 MiB by default), including an explicit
allowance for hash keys and allocator overhead. This follows Open3D's globally sparse,
locally dense [VoxelBlockGrid design](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/voxel_block_grid.html)
instead of allocating a room-sized dense volume.

The active submap rolls over before integration when travel reaches 2.5 m, accumulated
rotation reaches 100 degrees, 450 keyframes have been integrated, 82% of the sparse block pool
is active, or tracking relocalizes across a discontinuity. Completed volumes are extracted once
to compact host points and optional mesh geometry; only their rigid transforms remain mutable.
This is the same scalable principle used by Kähler et al.'s
[real-time submap reconstruction](https://www.robots.ox.ac.uk/~lav/Papers/kahler_etal_eccv2016/kahler_etal_eccv2016.html).

Host preview memory is also bounded: at most 64 submaps, 750,000 retained points, and 600,000
cached preview triangles. Older mesh caches are discarded before point guidance, and point
samples are deterministically compacted. Reaching the hard submap ceiling freezes integration
while tracking and raw archival continue.

Map publication runs independently from viewport rendering. Normal and tracking overlays
share bounded point snapshots, coverage is refreshed at a lower adaptive cadence, and mesh
extraction runs only at degradation level zero. The hysteretic pressure controller first lowers
publication rates, then pauses mesh/coverage work, and finally integrates fewer keyframes; it
never reduces archive fidelity or skips pose tracking.

## Live relocalization and loop correction

The tracker owns a 48-entry local anchor database. It retains the first accepted view, a
deterministically thinned capture-wide history, and the eight newest anchors; a rotating query
therefore searches the whole take without turning tracking loss into an unbounded GPU search.
Recovery still requires three temporally consistent, depth-verified observations, and neither
the lock frame nor any intermediate rejected pose is fused.

Completed submaps are nodes in a bounded rigid pose graph. Adjacent nodes are certain odometry
edges. On submap completion, at most three non-adjacent candidates are queried when the adaptive
controller has headroom. A candidate becomes an uncertain loop edge only after robust
point-to-plane ICP passes overlap, correspondence, metric residual, translation, and rotation
gates. ScanLan uses Open3D's documented
[multiway pose-graph model](https://open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html),
then rejects an optimized graph whose node correction or verified-loop residual exceeds the
live safety limits.

An accepted solve changes rigid submap transforms only: depth is not reintegrated and point
buffers are not duplicated. The viewport interpolates each correction for 350 ms, while a
separate map-to-odometry transform preserves active tracking continuity and places future
submaps in the corrected frame. The coverage field receives the same map-frame correction.
Every accepted and rejected loop decision is written asynchronously to `live_loops.jsonl` for
production revalidation. The final stop artifact settles the exact optimized transforms;
production still reruns tracking, verifies loops from raw RGB-D, and may reject every live
decision.
