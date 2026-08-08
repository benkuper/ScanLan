# Accelerated and hybrid reconstruction plan

Status: first implementation and the repository 4K HEVC video validation are complete on 2026-08-08; the wider release-quality scene matrix remains.

This document is the durable implementation plan and decision log for three related workstreams:

1. CUDA-enabled PyCOLMAP for NVIDIA systems.
2. A fast, bounded, observable CPU fallback.
3. A hybrid RGB-D plus photo/video reconstruction path.

Update this document when scope, architecture, thresholds, or rollout decisions change.

## Implementation status

- M0: implemented truthful worker-owned ETA, stage ETA propagation, and live decode/selection metrics; benchmark fixtures/report remain.
- M1: reproducible native/redistributable CUDA build implemented; native `sm_120` extraction/matching, gsplat, and production media-preparation paths validated.
- M2: full-core-minus-two CPU policy, streaming bounded-memory video selection, a 1 fps/240-frame default budget, batched cancellable CPU SIFT, bounded multi-model mapping, quality-gated partial-model recovery, native-stage heartbeats, and reusable media-observation cache implemented.
- M3: hybrid project/job orchestration, early RGB-D camera export, cached media observations, depth-backed media localization, and timestamp-ordered video spatial priors implemented. Joint metric bundle adjustment remains a follow-up.
- M4: schema-4 mixed datasets, optional per-frame depth, fixed RGB-D pose anchors, balanced sampling, high-quality appearance optimization, point recoloring, and shared mesh texture observations implemented. Quality benchmarking remains.
- M5: automated regression, production build, runtime dependency, CUDA smoke, one real-photo production-path solve, and the repository 180-second 4K HEVC video solve passed. Full hardware/scene matrix, cancellation/resume stress, and hybrid quality baselines remain.

## Outcomes

- Camera feature extraction and matching use CUDA when a packaged CUDA PyCOLMAP runtime passes a real smoke test.
- CPU reconstruction remains usable, consumes the available processor intentionally, and never retains an unbounded number of decoded full-resolution video frames.
- Progress represents measured work. An ETA is shown only when the active stage has enough throughput observations to support one.
- Projects may contain both RGB-D captures and media sources.
- Hybrid projects localize media cameras against depth-backed metric RGB-D landmarks, refine those cameras without losing metric scale, and expose the resulting high-resolution observations to point-cloud coloring, mesh texturing, and Gaussian training.

## Delivery milestones

### M0: Benchmarking and observability

- Add stage telemetry for decoded/selected frames, images, features, pairs, registered cameras, elapsed time, and backend.
- Separate current-stage progress/ETA from weighted whole-job progress.
- Remove whole-job ETA extrapolation during indeterminate stages.
- Add fixtures for short/long video, unordered photos, and overlapping RGB-D plus media.

Acceptance:

- The first video-decode progress update appears within two seconds.
- No increasing ETA is displayed while progress is indeterminate.
- Job records remain backward compatible.

### M1: CUDA PyCOLMAP

- Add a reproducible Windows source build for COLMAP/PyCOLMAP 4.1.1 using vcpkg and CUDA.
- Support a native developer build and a redistributable release build.
- Package repaired native dependencies with the splat worker.
- Validate CUDA with real SIFT extraction and matching, not only `pycolmap.has_cuda`.
- Select `Device.cuda` only after validation and fall back safely to CPU.

Acceptance:

- Packaged diagnostics report the PyCOLMAP version, CUDA build state, architecture, and validated feature backend.
- Feature extraction and matching visibly use the NVIDIA GPU.
- CUDA extraction/matching is at least three times faster than the CPU baseline on the RTX 5080 without losing more than five percentage points of registration ratio.

### M2: CPU fallback and media preparation

- Remove the eight-thread ceiling and reserve a small amount of CPU headroom.
- Stream video selection with bounded memory.
- Adapt image, frame, and feature budgets to source size and backend.
- Use sequential/loop-aware pairing for video and exhaustive matching only for small unordered photo sets.
- Batch cancellable feature extraction and cache completed preparation stages.

Acceptance:

- Decode memory is bounded independently of video duration.
- CPU extraction normally reaches at least 80% utilization when not I/O limited.
- Completed preparation stages are reusable after interruption.

### M3: Hybrid project and metric localization

- Permit RGB-D captures and media sources in the same project.
- Add an explicit `hybrid` source kind to artifact jobs.
- Extract spatially distributed RGB-D anchor views with depth-backed 3D landmarks.
- Generalize the existing supplemental-photo localization code into a reusable media localizer.
- Localize photos independently and video frames using temporal priors plus RGB-D relocalization.
- Run metric-anchored pose-prior bundle adjustment.
- Publish one shared, immutable media-observation set with calibrated cameras, poses, confidence, source timestamps, and normalized images.

Acceptance:

- Rejected media never enters a dataset.
- Accepted synthetic poses are within 2 cm and 1 degree.
- Metric scale drift remains below 1%.

### M4: Mixed dataset and hybrid Gaussian training

- Introduce canonical dataset schema 4 with per-frame source type, optional depth, pose confidence, and anchor state.
- Apply depth and normal losses only to frames that contain depth.
- Balance RGB-D and media sampling.
- Keep RGB-D poses strongly anchored while allowing bounded media-pose refinement.
- First release: metric 2D surface seeds with high-resolution media appearance optimization.
- Later extension: guarded free 3D Gaussians for media-observed geometry outside the RGB-D surface.
- Re-color the metric point cloud from localized high-resolution observations using depth-tested visibility, view-angle, distance, sharpness, and exposure weights.
- Use the same observations as preferred mesh-texture cameras, with RGB-D frames as coverage fallback.

Acceptance:

- Existing schema-3 datasets remain readable or are migrated deterministically.
- Hybrid holdout-view quality exceeds both the RGB-D-only and media-only baselines.
- Point-cloud and mesh outputs remain metric and RGB-D-derived.
- Point-cloud positions and mesh geometry remain metric and RGB-D-derived, while their published colors/textures may come from the localized high-quality camera.

### M5: Release validation

- Test CUDA, CPU fallback, missing-CUDA, cancellation, resume, cache invalidation, packaging, and corrupted-cache recovery.
- Record benchmark results in a versioned report under `docs/`.
- Update user-facing runtime/backend descriptions.

## Architectural decisions

### Backend selection

- `cuda`: explicitly validated CUDA SIFT extraction and matching.
- `cpu`: full-core CPU fallback with adaptive budgets.
- `auto`: prefer validated CUDA, otherwise CPU. Never silently label `auto` as CUDA.

### Progress and ETA

- Workers own stage progress and stage ETA because they observe actual work units.
- The desktop owns only weighted overall progress.
- An absent worker ETA remains absent; the desktop does not invent one from overall percent.
- Opaque monolithic calls show an indeterminate stage state unless they expose callbacks or can be safely batched.
- Native CUDA matching and CPU mapping publish a one-second elapsed-time heartbeat without inventing an ETA.

### Hybrid coordinate system

- RGB-D establishes world axes, origin, and metric scale.
- Media cameras are localized into that world through depth-backed 2D-to-3D correspondences.
- RGB-D poses are fixed during initial refinement and may only receive explicitly bounded corrections later.

### Hybrid representation

- Point-cloud positions and mesh geometry continue to use measured RGB-D geometry.
- Localized media is a shared appearance source: it re-colors metric points, drives the mesh atlas, and supervises splat appearance.
- RGB-D color remains a coverage fallback wherever no validated high-resolution observation sees a surface.
- The initial hybrid splat uses metric 2D surface seeds and media for appearance.
- Additional unconstrained 3D geometry is a separate guarded extension, not part of the first hybrid release.

## Benchmark matrix

- Video: 30 seconds, 2 minutes, and 5 minutes at 4K; H.264 and HEVC where available.
- Photos: 30, 100, and 300 images.
- Scene conditions: textured, low texture, motion blur, repeated structure, partial overlap, and lighting change.
- Hybrid: full overlap, partial overlap, and deliberately unrelated media.
- Hardware paths: validated CUDA, forced CPU, and CUDA-runtime failure fallback.

For every case record wall time, peak memory, utilization, selected image count, feature count, pair count, camera registration ratio, reprojection error, metric scale error, and final holdout quality.

## Decision log

- 2026-08-08: Implement all three workstreams; keep this plan as the canonical discussion record.
- 2026-08-08: Use COLMAP/PyCOLMAP 4.1.1 and CUDA 13.3. The local development target is Blackwell `sm_120`.
- 2026-08-08: Treat truthful progress as a correctness requirement, not a cosmetic improvement.
- 2026-08-08: Reuse and generalize the existing depth-backed supplemental-photo localizer for hybrid media.
- 2026-08-08: Hybrid media must improve all three outputs. Geometry remains depth-derived; point colors, mesh textures, and splat appearance consume the same localized high-resolution observation set.
- 2026-08-08: Hybrid schema 4 remains metric at the dataset level but allows depthless media frames; only RGB-D frames are metric pose anchors and depth-loss providers.
- 2026-08-08: Fixed RGB-D Gaussian positions/scales preserve measured geometry, while hybrid runs continue optimizing opacity and spherical-harmonic appearance at the requested iteration budget.
- 2026-08-08: Default video sampling is 1 fps with a 240-frame ceiling. Standalone SfM may publish a geometrically valid partial model after registering at least 25% of selected views; disconnected views are reported as a quality warning. Hybrid imports do not need this standalone solve and localize decoded observations directly against the metric RGB-D map.
