# High-quality RGB-D reconstruction

## Target architecture

ScanLan should keep two complementary metric representations:

```text
synchronized RGB-D + calibration
              |
      globally optimized poses
        /                 \
TSDF / surfel geometry     2D Gaussian appearance
        |                  |
point cloud + mesh          photorealistic splat
```

The TSDF/surfel path remains the authority for measurement, collision, point-cloud export, and mesh extraction. Surface-aligned 2D Gaussians provide view synthesis and presentation quality. A splat must not be treated as a substitute for a watertight geometric model.

This combines the strongest ideas from:

- [GSFusion](https://gs-fusion.github.io/): sparse TSDF-backed allocation, contrast-aware image subdivision, keyframe replay, and a short global appearance refinement.
- [2DGS-SLAM](https://github.com/PRBonn/2DGS-SLAM): surface-disc Gaussians, depth-consistent rendering, normal and distortion regularization, active/inactive local maps, and loop-closed poses.

No source from GSFusion is copied. Its repository includes a Gaussian-Splatting license that restricts the complete project to non-commercial use. ScanLan implements the published ideas on top of Apache-licensed gsplat.

## What the RTX 3060 result means

GSFusion reports the following mapping results on a desktop RTX 3060:

| Dataset | Mapping | GPU memory | Model size |
| --- | ---: | ---: | ---: |
| ScanNet++ | 6.14 fps | 2.81 GB | 29.3 MB |
| Replica | 9.73 fps | 7.26 GB | 40.1 MB |

Those experiments use ground-truth camera poses for all compared methods. The real drone demonstration obtains poses from OKVIS2. The reported FPS is therefore a mapping target, not an end-to-end capture, tracking, loop-closure, fusion, and export target.

Their quality-oriented ablations are directly useful to ScanLan:

- 1 cm TSDF/occupancy voxels are the quality baseline.
- Reducing the quadtree threshold from 0.1 to 0.01 improves ScanNet++ novel-view PSNR from 25.45 to 25.96 while reducing mapping speed from 6.14 to 5.09 fps.
- Ten post-scan passes over roughly 340 keyframes take about 60 seconds and improve novel-view PSNR from 22.76 to 25.45. Twenty passes take about 121 seconds and reach 25.87.
- Random historical-keyframe replay produces a large quality gain in the online setting without materially changing mapping speed.

## Implemented first slice

The ScanLan RGB-D splat path now uses:

1. Native gsplat 2DGS rasterization rather than volumetric 3DGS rasterization.
2. RGB-D quadtree initialization with a maximum leaf size so blank walls retain geometric coverage while high-contrast areas receive more discs.
3. One best seed per 1 cm world voxel, capped at 350,000 initial discs to keep startup and memory bounded on midrange GPUs.
4. Surface normals from local depth derivatives, stored as initial disc rotations.
5. Projected cell footprints as anisotropic tangent scales.
6. L1 + SSIM color loss, robust metric depth loss, rendered/depth normal consistency, and 2DGS distortion regularization.
7. Degree-three spherical harmonics with a progressive activation schedule.
8. Deterministically shuffled keyframe epochs to prevent trajectory-order bias.
9. Thin-disc conversion for the existing 3DGS-compatible PLY and realtime preview format.
10. Bounded per-keyframe pose refinement for metric RGB-D jobs after a geometry warm-up. The first camera fixes the gauge, corrections are smoothed within each phase, and the refined trajectory is exported as `room-splat-cameras.json`.

The TSDF point cloud and mesh pipeline is unchanged.

## Next quality gates

### 1. Measure before changing tracking

Create a fixed RGB-D benchmark containing a short loop, blank walls, thin furniture, reflective glass, and a relocalization event. Record:

- trajectory ATE/RPE when a reference trajectory is available;
- mesh accuracy/completeness and TSDF-to-depth residual;
- held-out PSNR, SSIM, and LPIPS;
- peak VRAM, initial/final Gaussian count, iterations per second, and wall time;
- visible floaters, edge halos, holes, and texture seams.

Compare the previous 3DGS checkpoint and the new 2DGS checkpoint from the exact same posed-frame dataset.

### 2. Measure pose refinement on real captures

The bounded joint refinement is implemented. It:

- optimizes rigid camera corrections without a scale degree of freedom;
- keeps robust metric depth active alongside photometric loss;
- anchors the first pose and smooths adjacent corrections within each phase;
- caps translation corrections at 5 cm and exports correction statistics.

Compare held-out image/depth residuals and duplicated-surface artifacts against the unrefined input poses before expanding the correction limits.

### 3. Add global loop closure

For room-scale quality, pose drift will eventually dominate rasterizer quality. Add candidate retrieval and geometric verification, then optimize the pose graph. MASt3R can be an optional high-quality relocalizer, while a lighter image-retrieval plus RGB-D/ICP verification path should remain the default.

### 4. Add local-map optimization for live feedback

The final offline pass can optimize all selected keyframes. A live path should instead render and update only visible/active discs, replay a small random set of historical keyframes, and freeze distant discs. Apply loop-closure corrections to disc positions and rotations before a short global refinement.

## Acceptance target

The published RTX 3060 result is a useful lower-bound reference, not a deployment constraint. The practical ScanLan target is:

- live mapping preview at 5 fps or better at a training resolution near 960 px;
- no more than 8 GB peak VRAM in the default quality mode;
- a compact 150k-500k disc map before optional densification;
- a one-to-three-minute final appearance refinement for a room-scale capture;
- geometric accuracy determined by the RGB-D pose/TSDF benchmark, not by visual PSNR alone.

These are product targets to validate on ScanLan captures, not claims inherited from the reference papers.
