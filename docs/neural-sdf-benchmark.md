# P11 neural-SDF production surface

## Decision

ScanLan uses a small geometry-supervised neural SDF implemented inside the existing pinned
PyTorch/CUDA splat runtime. It is an optional **Max Quality** surface refiner, not a new source of
camera poses or unbounded geometry. The validated TSDF or learned dense mesh remains the fallback
and the authority for topology.

The strongest mature photometric alternatives were not suitable as the packaged default:

| Candidate | Strength | Concrete incompatibility |
|---|---|---|
| [Neuralangelo](https://github.com/NVlabs/neuralangelo) | High-fidelity hash-grid SDF with progressive levels and numerical gradients | Upstream says its default configuration requires at least 24 GB VRAM and reduced 12 GB settings sacrifice quality; its NVIDIA research license requires separate business licensing and is incompatible with ScanLan's default distributable runtime. |
| [SDFStudio](https://github.com/autonomousvision/sdfstudio) | Apache-2.0 NeuS/VolSDF/Neural RGB-D framework with Neuralangelo variants | The upstream frozen stack targets Python 3.8, PyTorch 1.12.1, and CUDA 11.3, conflicting with ScanLan's pinned Python 3.11, PyTorch 2.12, CUDA 13 Blackwell runtime. Shipping a second large CUDA dependency graph would also break the one-model-at-a-time 12 GB policy. |
| [instant-ngp](https://github.com/NVlabs/instant-ngp) | Fast Windows hash-grid SDF and mesh extraction | Its NVIDIA license limits use to noncommercial research/evaluation, and its SDF path trains from an existing mesh, so it would reproduce the same surface rather than use ScanLan's validated camera/depth evidence; integrating another C++/CUDA/OptiX runtime would not justify that limited role. |
| [Neural RGB-D Surface Reconstruction](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) | Metric RGB-D neural TSDF/SDF supervision | The original TensorFlow/Conda implementation is an old research stack and takes hours on its published comparisons. Its geometry-supervised principle fits ScanLan, but its runtime does not. |

The selected implementation keeps the compatible parts of the literature: metric signed samples
from validated geometry, progressive frequency activation, a gradient-norm Eikonal constraint, a
continuous zero level set, and held-out error measurement. It deliberately excludes camera/pose
optimization and photometric appearance from P11 because P9 already owns camera solving and P13
owns radiometric/material foundations.

## Pipeline and gates

1. Camera, scale, depth, multi-view, and free-space validation must report `accepted`.
2. The indexed production surface is normalized by a robust 1st–99th percentile bounding box.
3. Stable oriented vertices produce zero and signed ±0.45/±1.1 voxel training samples.
4. A deterministic 1/17 split is withheld from optimization.
5. A 128-wide Softplus MLP fits seven progressively enabled Fourier levels; every fourth step adds
   an automatic-differentiation Eikonal loss.
6. Original vertices are projected toward the learned zero set with a hard two-voxel step ceiling.
7. The CUDA worker rejects excessive held-out error, displacement, flipped triangles, or new
   degeneracy.
8. The reconstruction worker independently verifies byte-safe arrays, identical topology, finite
   coordinates, and displacement limits.
9. Only an accepted candidate reaches depth-aware repair and multiview texturing. All other outcomes
   retain the baseline and publish an explainable report.

The content-addressed cache fingerprint includes the implementation revision, source vertices,
triangles, and metric voxel size. Cached geometry is independently checked again when loaded.

## Verification in this phase

Automated tests cover small accepted motion, excessive displacement, flipped orientation, and the
camera-validation fail-closed path. A direct CUDA optimizer smoke test on the reference NVIDIA
GeForce RTX 5080 Laptop GPU (compute capability 12.0) used a noisy 8×8 plane (64 vertices, 98
triangles, 200 iterations) and completed with:

| Metric | Result |
|---|---:|
| Held-out SDF MAE | 0.46 mm |
| Held-out SDF p95 | 1.10 mm |
| Median vertex displacement | 0.22 mm |
| p95 vertex displacement | 0.66 mm |
| Flipped / degenerate triangles | 0 / 0 |
| Gate | accepted |

The complete cross-process path was also exercised at the production 1,600 iterations: immutable
input NPZ, CUDA subprocess, atomic candidate/report, and independent reload validation all passed;
the independent checker measured 0.19 mm median and 0.40 mm p95 displacement.

This smoke test proves the optimizer and acceptance path execute; it is not a representative room
quality benchmark. A real-capture A/B remains required before enabling the option by default. The
release gate is improved held-out surface/alignment error without loss of openings, thin structures,
or texture coverage on the 12 GB reference GPU. Until that evidence exists, P11 remains opt-in.
