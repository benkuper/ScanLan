# ScanLan validation engine

P6 introduces `scanlan-validation`, a NumPy-only package shared by the realtime/production
RGB-D worker and the isolated media/geometry pipeline. Model execution is not evidence that an
output is safe: every backend must pass the same versioned gates before its cameras, depth, or
points can enter a reconstruction.

## Contract

Contract version 1 provides five independent, serializable validators:

| Gate | Evidence | Fail-closed result |
|---|---|---|
| Camera | finite rigid SE(3), proposal confidence, robust translation continuity, angular continuity | reject the unsafe frame and freeze its geometry |
| Scale | paired 3D metric anchors, robust similarity fit, inlier ratio, relative RMSE | retain an explicit unverified/relative label |
| Depth | overlap with trusted metric depth, residual percentiles, scale bias, inlier ratio | retain measured depth unchanged |
| Free space | proposed and observed depths on the same calibrated ray | reject geometry in front of an observed surface |
| Geometry | finite bounded coordinates, confidence, observation support, free-space mask | discard inadmissible points; reject an empty result |

Camera translation limits are data-driven from the trajectory median and median absolute
deviation. Motion is normalized by the caller's source frame or timestamp position, so adaptive
keyframe spacing is not mistaken for a pose jump. Callers may also impose a stricter physical or
model-unit ceiling. Production RGB-D cameras have already passed metric odometry and pose-graph
gates, so their final sanity pass uses a 0.5 m-per-source-frame ceiling instead of learning a new
speed distribution from sparsely selected keyframes. Scale is recovered
with a proper-rotation SVD similarity fit and iteratively robust residual trimming, based on
[Umeyama's least-squares formulation](https://doi.org/10.1109/34.88573). A successful model
inference never upgrades scale by itself; only metric correspondences can produce
`MODEL_METRIC_VALIDATED`.

The free-space classifier follows the line-of-sight semantics of Curless and Levoy's
[volumetric range integration](https://graphics.stanford.edu/papers/volrange/): a proposal closer
than a trusted measured surface occupies already observed empty space and is a contradiction.
A point behind the surface is merely occluded and does not independently validate or invalidate
the proposal. TSDF fusion still consumes only cameras and depths that survive these gates, as in
Open3D's documented [integration pipeline](https://open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html).

## Current integrations

- Progressive LingBot video submaps use shared camera and geometry masks and expose the latest
  validation report alongside confidence, drift, and scale telemetry.
- Production RGB-D cameras are validated independently within each capture phase. Rejected poses
  are removed before depth refinement, dataset publication, or fusion; cross-phase placement is
  evaluated separately by registration.
- LingBot depth completion uses the shared metric depth gate. Multi-view confirmation also tracks
  explicit free-space contradictions, and a contradicted hole pixel cannot be generated.
- MapAnything RGB-D completion reverses model preprocessing to the sensor grid, fits local metric
  residuals on sensor anchors, and applies the shared depth gate only to a deterministic held-out
  anchor set. Photo and short-video camera/depth proposals pass the same camera/geometry gates and
  must agree with COLMAP before becoming a dense prior.
- DA3 Nested runs camera/depth proposals in bounded overlapping windows. Every overlap must pass
  center and rotation continuity gates; the assembled proposal must then beat the accepted baseline
  on normalized COLMAP camera residual. Pose-conditioned RGB-D depth and direct Gaussian seeds pass
  the same shared geometry, metric, multiview, and free-space gates before publication.
- Frozen worker builds install this package before packaging either runtime; PyInstaller discovers
  the ordinary imports, so no runtime network access is needed.

Every later backend must return proposals through this contract. Backend-specific
confidence can tighten a gate but cannot weaken the common minimum evidence or silently relabel
scale.

## Verification

Run the package tests with either project runtime after installing the local package:

```powershell
splat-worker\.venv\Scripts\python.exe -m pip install --no-deps .\validation
splat-worker\.venv\Scripts\python.exe -m unittest discover -s validation\tests -v
```

Tests cover malformed/discontinuous cameras, robust similarity recovery with an outlier, metric
depth scale rejection, ray support/free-space/occlusion classification, and fail-closed point
filtering. Worker integration tests protect production phase boundaries and hole-only depth
completion.

The retained physical Femto/LingBot depth run provides a representative gate check: all 267
validated production cameras passed, with a maximum 0.141 m selected-keyframe step and 13.52
degree rotation. Twenty sampled learned-depth frames all passed metric agreement (12.38-19.48 mm
median residual and 0.9138-0.9562 inlier ratio), while a deliberate 20% scale perturbation was
rejected. These figures are evidence for this capture, not universal model-accuracy claims.

The P7 MapAnything check used three separated frames from the same physical Femto Mega capture.
The raw proposals were correctly rejected at 83.7-98.9 mm median residual. Held-out sensor-anchor
calibration then passed all three frames at 2.75-3.25 mm median residual; the independent-view and
free-space gates accepted 15,462 original-hole pixels (2.294% of reliable measured support). A
frozen BF16 runtime repeated model inference on the RTX 5080 with offline flags and an empty Torch
cache. These measurements validate the adapter and gates on this capture, not every scene.

The P8 DA3 check used real frames 0, 5, and 10 from the managed `Terrasse` capture. The rebuilt
frozen worker accepted all three cameras and published 370,992 finite direct Gaussians with exactly
370,992 learned opacity values (range 0.000002-0.995733), positive scales, and unit quaternions.
An SH0-plus-opacity gsplat render was visually inspected against the source frame and retained the
room structure and appearance; an opacity-free point projection was explicitly rejected as an
invalid way to judge a direct-GS representation. A deliberately incoherent 0/50/100 frame set was
rejected with mean camera confidence 0.1314 and no accepted cameras. Frozen diagnostics measured
7,881.1 MiB peak CUDA allocation against the 11,264 MiB safety limit and exercised both the
pose-conditioned depth and direct-Gaussian heads offline. These are capture-specific integration
measurements, not universal reconstruction-accuracy claims.
