# Adaptive backend policy

P18 adds one versioned, quality-first selector shared by the live RGB-D, production RGB-D,
media-camera, surface, and Gaussian lanes. It does not treat an installed model, a CUDA build, or a
successful import as proof that a backend is suitable for a capture.

## Selection contract

`scanlan_validation.backend_policy` accepts four independent inputs:

- source characteristics: RGB-D/photo/video/hybrid mode, sensor family, measured frame count,
  maximum image dimension, and declared capture characteristics;
- hardware evidence: a completed CUDA kernel smoke test, GPU identity and compute capability,
  total/currently-free VRAM, and CPU thread count;
- runtime evidence: availability, real smoke-test status, and pinned code revision for each
  candidate;
- quality intent and overrides: maximum-quality ordering, commercial-use constraints, and any
  explicit user backend choice.

Every benchmark record declares a bounded source and hardware envelope, required runtime
revisions, legal-use status, release gates, peak memory, and lane-specific task metrics. A record is
eligible only when all of those facts match. Total VRAM must include the record's measured peak plus
its safety reserve, while currently free VRAM must cover the working set plus launch headroom.
PyTorch's `cudaMemGetInfo` wrapper supplies free/total memory in the isolated learned runtime, and
the live worker uses NVIDIA's selective `nvidia-smi` query after Open3D has executed its CUDA probe.

Automatic ranking is lexicographic and quality first rather than a tunable weighted score:

- live RGB-D: accepted-frame ratio, pose p95, map-publication p95, then memory;
- live video: drift risk, accepted-frame ratio, first-geometry latency, then memory;
- production cameras: registration coverage, learned/COLMAP camera residual, reprojection error,
  then runtime;
- depth completion: held-out metric residual, inlier ratio, accepted-hole coverage, then memory;
- surfaces: held-out error, bounded displacement, then runtime;
- Gaussians: raw-render PSNR, SSIM, L1, then memory.

Missing metrics sort behind complete evidence. Failed gates, mismatched revisions, unvalidated CUDA,
an incompatible source, insufficient memory, and license conflicts reject a record rather than
receiving a penalty that another metric could hide.

## Operational integrations

At live-engine startup, ScanLan first executes the established Open3D CUDA/CPU probe. The policy
then audits that real result against the sensor and benchmark manifest and writes
`<phase>/backend-policy.json`. With incomplete evidence it keeps the current guarded Open3D path;
an explicit `--device cpu|cuda` remains an explicit, visibly unbenchmarked override.

Photo/video preparation asks the isolated geometry worker to smoke-test applicable model runtimes,
matches the selected observation set and active hardware, and writes
`outputs/backend-policy.json`. A compatible camera decision controls which learned proposal leads
the solve. Every proposed edge still passes ALIKED/LightGlue or SIFT matching, COLMAP geometric
verification, global bundle adjustment, and learned-camera agreement. The protected fallback is
the existing multi-challenger bake-off, so a missing policy cannot weaken camera quality.

RGB-D Depth refinement now has an **Adaptive · benchmark gated** option. It validates the isolated
LingBot-Depth, MapAnything, and DA3 runtimes, then selects only from compatible depth benchmarks.
No compatible record means sensor-only depth. The existing explicit backend options are unchanged;
all of them still preserve measured depth and pass metric, multiview, RGB-coverage, and free-space
gates.

The source-controlled manifest contains only the narrow physical measurements already recorded in
the P3/P6/P7/P9 reports. Characteristics not established by those captures are deliberately absent,
so the selector does not extrapolate a short static smoke or one phone-photo set into a universal
default. P19 remains responsible for the complete release matrix and default changes.

## Reproducibility and extension

The built-in schema-1 manifest is packaged into every frozen worker. For a controlled benchmark
campaign, set `SCANLAN_BACKEND_BENCHMARKS` to an absolute schema-1 manifest. Its content digest is
part of media dataset cache identity, so changing benchmark evidence cannot silently reuse a camera
dataset built under another policy. A policy report records the manifest revision, normalized
inputs, every selected/fallback decision, and the rejection reasons for every candidate.

`benchmark-live` embeds the policy report in its canonical output, allowing a measured run to retain
the exact decision that produced it. Candidate records still need review and explicit addition to a
manifest; benchmark output is never self-promoting.

Relevant upstream capability interfaces:

- [PyTorch `torch.cuda.mem_get_info`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html)
- [PyTorch `torch.cuda.get_device_properties`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_properties.html)
- [NVIDIA `nvidia-smi` selective queries](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
