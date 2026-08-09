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

Both runtimes and their model assets are verified during packaging and remain available offline.
The implementation follows the upstream
[LingBot-Map streaming API](https://github.com/Robbyant/lingbot-map) and the recommended
[LingBot-Depth v0.5 metric model](https://github.com/Robbyant/lingbot-depth). ScanLan retains
its stricter confidence, metric, free-space, multiview, and production-camera gates outside the
model process.

## Commands

```text
scanlan-geometry.exe diagnostics [--require-lingbot] [--require-lingbot-depth] [--require-flashinfer]
scanlan-geometry.exe infer-lingbot-map --request REQUEST.json --progress PROGRESS.json
scanlan-geometry.exe refine-rgbd-depth --request REQUEST.json --progress PROGRESS.json
```

Diagnostics loads the pinned assets when required. It also executes the real FlashInfer paged
attention shape and a BF16 LingBot-Depth inference on the installed CUDA device; package
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
