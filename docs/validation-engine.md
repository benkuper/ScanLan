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
- Frozen worker builds install this package before packaging either runtime; PyInstaller discovers
  the ordinary imports, so no runtime network access is needed.

Later MapAnything and DA3 adapters must return proposals through this contract. Backend-specific
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
