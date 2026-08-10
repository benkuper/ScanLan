# P12 Gaussian initialization and production optimization

## Decision

ScanLan retains the pinned `gsplat==1.5.3` CUDA runtime and generalizes the data contract and
training schedule around it. A schema-1 manifest now identifies each starting representation:

| Kind | Source | Representation | Optimization policy |
|---|---|---|---|
| `sparse_sfm` | geometrically verified COLMAP tracks | volumetric 3D Gaussians | adaptive densification |
| `dense_surface` | calibrated RGB-D or validated learned depth | oriented 2D discs for metric data, bounded 3D surface prior for media | fixed metric surface or bounded refinement |
| `direct_gaussian` | accepted DA3 direct head | predicted anisotropic 3D Gaussians | preserve rotation, scale and opacity, then bounded refinement |

The official [3D Gaussian Splatting paper and implementation](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
establish sparse SfM initialization and adaptive density control. The official
[2D Gaussian Splatting implementation](https://github.com/hbb1/2d-gaussian-splatting) uses oriented
surface discs with depth-distortion and normal regularization for geometric reconstruction. The
[gsplat rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html) documents packed
rasterization as the bounded-memory path and exposes pixel-calibrated pinhole intrinsics. P12 keeps
those proven roles instead of forcing all inputs into one seed type.

The current pinned runtime's `DefaultStrategy` remains the production densifier. Newer gsplat
[MCMC densification](https://docs.gsplat.studio/main/apis/strategy.html) is promising for sparse
photometric reconstruction, but relocating low-opacity samples is not an acceptable default for
ScanLan's measured surface seeds or a direct model's predicted primitives. Changing strategy and
runtime revision together would also make this phase's initialization comparison impossible to
attribute. A strategy bake-off remains appropriate for the later adaptive-backend phase.

## Source-resolution schedule

Allocating a complete 2560 px or 4K raster contradicts the 12 GB safety target. Permanently resizing
those images to 960 px, however, throws away the source detail P12 is intended to optimize. The
trainer therefore uses two stages:

1. bounded global views establish coverage, low-frequency appearance, geometry and camera updates;
2. the final schedule visits every source-camera tile at native pixel density.

A source tile retains `fx` and `fy`; subtracting its integer left/top origin from `cx` and `cy`
produces exactly the same calibrated rays as the corresponding pixels in the complete image. Tile
width and height never exceed the hardware raster tier. The final-stage length is adaptive: it is at
least 20% of the requested optimization and grows when the measured camera/tile count requires more
iterations. If necessary, the effective iteration count is extended so every tile is seen once.

## Quality and safety gates

* A dense contract without its parameter sidecar is rejected.
* A direct contract without independent predicted opacity is rejected.
* Kind, representation, direct-prior flag and sidecar path must agree.
* Confidence and opacity must be finite, bounded arrays with one value per seed.
* Metric 2DGS remains fixed and cannot be silently converted to volumetric 3DGS.
* Every expected source tile must be observed before final PLY publication.
* Five deterministic calibrated training views are rendered without the optimizer's per-view
  exposure transform, matching what an interoperable PLY viewer receives. Publication requires
  median PSNR ≥18 dB, median SSIM ≥0.55, and median L1 ≤0.15.
* A photometric rejection writes `outputs/splat-quality-report.json` and atomically preserves the
  final optimizer checkpoint so a longer run can resume rather than restart.
* The hardware-specific Gaussian ceiling, pre-growth stop, pinned-host LRU and atomic checkpoint
  behavior remain unchanged.
* Checkpoints preserve the per-camera tile counters so resume continues the same deterministic pass.

## Verification in this phase

The splat-worker suite covers all three manifest kinds, fail-closed contract disagreement,
confidence/opacity separation, deterministic complete tile scheduling, crop bounds, exact focal
length and principal-point adjustment, RGB source-pixel selection, existing 12 GB Gaussian limits,
checkpoint atomicity, pose constraints, dense/direct scale behavior, and all learned-backend dataset
paths. A real CUDA scene benchmark remains required before changing iteration defaults or comparing
densification strategies; the training sidecar now records the evidence needed for that comparison:
requested/effective iterations, initialization kind, representation, source-resolution start,
observed/expected tile count and coverage ratio.

The P12 integration smoke used a real nine-view, 1591×894 indoor photo solve on the RTX 5080 Laptop
GPU. The trainer grew 797 verified sparse tracks to 1,663,627 bounded Gaussians, stopped growth before
the three-million ceiling, survived an interrupted-command resume from its 1.19 GB atomic checkpoint,
and completed all 1,000 scheduled source-resolution steps. Visual inspection still found severe
oversized and over-saturated splats at 5,000 iterations. The raw five-view evaluation measured median
14.22 dB PSNR, 0.488 SSIM, and 0.144 L1, so the new gate correctly rejects that diagnostic candidate
instead of treating complete execution or 100% tile coverage as proof of quality.
