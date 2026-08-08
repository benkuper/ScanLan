# Hybrid reconstruction validation record

Date: 2026-08-08

## CUDA PyCOLMAP smoke and microbenchmark

Hardware/runtime:

- NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0 (`sm_120`)
- CUDA toolkit 13.3
- PyCOLMAP/COLMAP 4.1.1, native CUDA build
- Python 3.11; CPU fallback configured for 22 workers on 24 logical CPUs

The required runtime diagnostic executed a real CUDA SIFT extraction and match plus the gsplat 2DGS forward/backward smoke test. It reported 562 features, 562 self-matches, and `cudaValidated: true`.

The feature microbenchmark used three deterministic 1600 × 1200 textured grayscale images and an 8,192-feature budget:

| Backend | Extraction | Features | Self-match | Matches |
| --- | ---: | ---: | ---: | ---: |
| CPU | 2.459 s | 15,117 | 1.624 s | 4,994 |
| CUDA | 0.118 s | 14,553 | 0.081 s | 4,828 |

- Extraction speedup: 20.84×
- Matching speedup: 20.05×
- CUDA/CPU feature-count difference: 3.73%

This is a runtime/backend microbenchmark, not an end-to-end registration-quality benchmark. The video/photo/hybrid scene matrix and registration-ratio gates in the canonical plan remain release-validation work.

## Production-path media smoke

The production `prepare-media` command processed the repository's 16 overlapping phone-photo fixtures at a 1,600-pixel image limit. CUDA SIFT extraction, exhaustive CUDA matching, incremental mapping, undistortion, sparse initialization, and immutable dataset publication completed in 4.454 seconds.

- Feature backend: `COLMAP SIFT GPU`
- Registered cameras: 9/16 (56.25%); the solver reported two disconnected models and published the larger one with an explicit warning.
- Published sparse points: 797
- Median reprojection error: 0.639 px
- Mean reprojection error: 0.749 px

This confirms the production CUDA path and quality reporting. The fixture does not meet the preferred 85% single-model registration ratio, so it is not used as a release-quality acceptance scene.

## Repository 4K HEVC video

The production `prepare-media` source path processed `test-photos/A001_08081059_C002.mp4`: a 180.300-second, 5,408-frame, 3840 x 2160 HEVC capture at 29.995 fps. The bounded default selected 181 sharp keyframes at 1 fps and completed dataset publication in 199.194 seconds (3m19s).

- Feature backend: `COLMAP CUDA SIFT`; observed GPU utilization reached 64% during matching.
- Incremental mapper: 22 CPU workers, at most five recovery models, a 226-second hard budget, and a live elapsed-time heartbeat.
- Registered cameras: 85/181 (46.96%) in the largest of four disconnected models.
- Published reliable sparse points: 35,601.
- Median reprojection error: 0.720 px; mean reprojection error: 0.790 px.
- Mean sparse-track length: 3.392 observations.
- Result: successful immutable schema-3 dataset publication with an explicit disconnected-view warning.

Two tuning probes informed the final policy. A forced single-model run selected an accidental five-frame opening fragment and failed in 67.7 seconds. Limited multi-model recovery at 1 fps reconstructed 85-86 cameras in about 3m20s; increasing density to 239 views reached 114 cameras but consumed the full five-minute mapper budget and preserved essentially the same registration ratio. The faster 1 fps default therefore provides the better balanced fallback for this clip. Hybrid RGB-D projects bypass standalone SfM and localize all selected media observations against the metric RGB-D map.

## Automated validation

- Reconstruction worker: 90 tests passed.
- Splat/media worker: 34 tests passed.
- Rust library/job orchestration: 30 tests passed.
- Svelte/TypeScript diagnostics: 0 errors and 0 warnings; production Vite build completed.
