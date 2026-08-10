# Learned geometry worker

Reconstruction 2.0 runs learned geometry models in `scanlan-geometry.exe`, separate from camera
capture, Open3D reconstruction, COLMAP media solving, and Gaussian optimization. The isolation
prevents a loaded foundation model or stale CUDA cache from sharing the critical capture process
or remaining resident while a later large model starts.

## Backends and pins

| Backend | Code revision | Model revision | Role |
| --- | --- | --- | --- |
| LingBot-Map long | `1f480aeb8a47a24656090d46d053115b7fe60435` | `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9` | Ordered streaming camera, depth, confidence, and dense seeds |
| LingBot-Depth v0.5 | `f3a237e434ae987bc38281476d6cfb5df3e4d739` | `79204ed6b837f4fdd192cf563e59481fecfa0295` | Metric RGB-D refinement and hole proposals |
| MapAnything Apache | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a` | RGB-D completion and bounded photo/short-video camera-depth challenger |
| DA3 Nested Giant-Large 1.1 | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `b2359bdf726fb44ef62acca04d629dcf158053e7` | Best-quality any-view camera/depth challenger, pose-conditioned metric depth, bounded streaming, and direct Gaussian proposals |

Both runtimes and their model assets are verified during packaging and remain available offline.
The implementation follows the upstream
[LingBot-Map streaming API](https://github.com/Robbyant/lingbot-map) and the recommended
[LingBot-Depth v0.5 metric model](https://github.com/Robbyant/lingbot-depth), the
[Apache MapAnything checkpoint](https://huggingface.co/facebook/map-anything-apache), and the
[refreshed DA3 Nested checkpoint](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1).
DA3's source is Apache-2.0 but its strongest checkpoint is CC BY-NC 4.0, so the UI and dataset
manifest preserve a noncommercial-use warning. ScanLan retains
its stricter confidence, metric, free-space, multiview, and production-camera gates outside the
model process.

## Commands

```text
scanlan-geometry.exe diagnostics [--require-lingbot] [--require-lingbot-depth] [--require-mapanything] [--require-da3] [--require-flashinfer]
scanlan-geometry.exe infer-lingbot-map --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe infer-mapanything --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe infer-da3 --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe refine-rgbd-depth --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe refine-rgbd-depth-mapanything --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe refine-rgbd-depth-da3 --request REQUEST.json --progress PROGRESS.json
```

Diagnostics loads the pinned assets when required. It also executes the real FlashInfer paged
attention shape plus BF16 LingBot-Depth, MapAnything, DA3 pose-conditioned depth, and DA3
direct-Gaussian inference on the installed CUDA device; package
discovery alone is not considered acceleration validation.

## LingBot-Map IPC contract

Request schema 1 contains only immutable paths and bounded scalar configuration:

- ordered image paths;
- optional normalized calibrated-ray array;
- increasing output-frame indices;
- maximum dense-seed count;
- result, array, progress, and cancellation paths.

The result is a compressed, lossless NumPy archive containing camera transforms, intrinsics,
points, RGB8 colors, scales, quaternions, source-frame ownership, and per-frame confidence.
JSON metadata records backend, model path, processed image size, counts, and the exact code and
model revisions plus model digest. The client rejects a worker built against any other pinned
revision. Publication uses a sibling temporary file plus atomic replacement. The client copies
arrays out of the archive, validates the complete contract, and removes transient IPC data only
after a successful read.

Progress is a latest-wins atomic JSON file. Cancellation is fail-closed: the parent sets the
shared flag, the child checks it at model progress boundaries, and the supervised Windows Job
Object terminates descendants if the owning artifact job exits unexpectedly.

When request preview paths are present, the same causal inference publishes completed frame
chunks as bounded local submaps. Preview arrays use a separate atomic NumPy contract and cannot
replace the final arrays. The caller caps the display map, validates finiteness, and preserves the
mandatory `MODEL_METRIC_UNVERIFIED` label. See [rgb-video-preview.md](rgb-video-preview.md).

## Equivalence and ownership

The geometry worker calls the same pinned `infer_lingbot_geometry` and `refine_depth_request`
implementations used before P4. The migration changes process ownership, not model math or file
formats. Automated equivalence tests round-trip every LingBot-Map output array bit-for-bit and
route the existing LingBot-Depth schema unchanged. Camera agreement, depth validation, fusion,
and final output publication remain in their previous owning workers.

## MapAnything contract

The pinned Apache model is loaded from a verified local `config.json` and
`model.safetensors`. Construction redirects UniCeption's Torch Hub call to MapAnything's bundled
DINOv2 source with `pretrained=False`; the complete checkpoint then supplies the weights. Frozen
diagnostics run with an empty Torch cache and offline Hugging Face flags.

RGB-D requests publish source-aligned depth, validity, and confidence arrays in bounded windows.
Image-only requests publish the same typed camera/dense-seed archive as LingBot, with an explicit
`MODEL_METRIC_UNVERIFIED` scale label. The media solver aligns both learned proposals to COLMAP
and uses the accepted lower-residual candidate; failed challengers cannot replace the baseline.

## DA3 contract

The pinned DA3NESTED-GIANT-LARGE-1.1 checkpoint keeps FP32 weights and uses BF16 safe autocast
(FP16 on older CUDA devices), matching the upstream heads that explicitly disable autocast. It
never shares a process with capture or gsplat optimization. Photo and video inference uses
24-frame windows with six-frame overlap. Successive windows are Sim(3)-aligned from overlapping
camera centers and orientations; a window is rejected above 0.12 normalized center residual or
8 degrees. Published telemetry records the window count, overlap, residual, rotation error, peak
CUDA allocation, and the direct-Gaussian decision.

The direct Gaussian head supplies means, SH0 color, learned opacity, positive anisotropic scales,
and world-space quaternions through the same validated initialization sidecar consumed by gsplat.
ScanLan preserves opacity instead of interpreting it as confidence, and does not overwrite the
head's learned local scale axis. A resolution-relative border crop mirrors the upstream exporter;
malformed splats are rejected by the shared geometry contract. Bounded windows keep
its resident input size independent of capture length. On a measured CUDA allocation failure the
isolated worker clears the failed allocation, records explicit fallback telemetry, and retries with
confidence-gated DA3 depth seeds to preserve 12 GB headroom. RGB-D refinement
passes accepted metric poses and intrinsics into DA3, then subjects its output to ScanLan's held-out
sensor-anchor, multiview, free-space, and provenance gates. Model metric scale never bypasses those
independent checks.

The Nested checkpoint applies its metric scale to depth and camera translations after its Gaussian
adapter runs. ScanLan applies the returned scale factor to Gaussian means and scales at the IPC
boundary, then requires the corrected cameras and geometry to pass the shared contracts before use.
