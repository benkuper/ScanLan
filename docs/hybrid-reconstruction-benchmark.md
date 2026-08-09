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

## Realtime RGB-D baseline

Reconstruction 2.0 protects the live path with a reproducible archive-replay benchmark. It
replays the original ordered RGB-D observations through the production realtime executable,
paces them from device timestamps by default, and records:

- pose, point-map, and mesh publication latency (median, p95, and maximum);
- processed, accepted, rejected, and integrated frame counts;
- source, tracking, mapping, and journal queue drops;
- tracking-state and relocalization counts;
- peak worker working set and NVIDIA process memory when available;
- final point/triangle counts and post-stop provisional/journal availability;
- the fail-closed invariant that no rejected frame is marked integrated.

Run the canonical benchmark against a representative capture with:

```powershell
.\build\worker-venv\Scripts\python.exe -m scanlan.cli benchmark-live `
  C:\path\to\capture-phase --mode mesh --device cuda `
  --session build\live-baseline-session `
  --report build\live-baseline.json
```

Use `--no-realtime-pacing` as a separate overload test. It is not comparable to the
interactive latency result because it deliberately feeds the engine faster than the sensor.
Machine- and scene-specific results belong in dated benchmark records; acceptance decisions
must not be inferred from synthetic fixtures alone.

### 2026-08-09 physical Femto baseline

The first Reconstruction 2.0 baseline replayed the existing 81-frame physical Femto Mega
capture `femto-live-smoke-6c9808ce` at its recorded 9.99 fps through the CUDA mesh profile on
the RTX 5080 Laptop GPU. This short, mostly static smoke capture validates the measurement
path and freezes current behavior; it is not the room-scale release scene.

| Metric | Baseline |
| --- | ---: |
| Pose latency | 31.0 ms median / 46.25 ms p95 |
| Point-map publication latency | 47.0 ms median / 718.7 ms p95 |
| Mesh publication latency | 70.5 ms median / 1,214.25 ms p95 |
| Accepted / integrated | 71 / 15 frames |
| Final provisional geometry | 59,263 points / 105,468 triangles |
| Tracking / mapping / journal drops | 10 / 0 / 0 |
| Peak worker working set | 923.33 MiB |
| Post-stop point packet / tracking journal | available / available |

The 160 reported source gaps are archive-rate sequence gaps from replaying a 10 fps archive
that was captured from a roughly 30 fps sensor stream; they are not benchmark-induced archive
loss. Per-process NVIDIA memory was unavailable through `nvidia-smi` under the active Windows
driver, so this run does not establish the VRAM baseline. The high first-publication outliers
also show why P1/P2 must separate warm-up, map-update cadence, and steady-state latency rather
than relying on tracking fps alone.

### 2026-08-09 P2 bounded-live-map revalidation

The same physical archive and CUDA mesh configuration were replayed alone after the P2
submap, budget-controller, coverage, and confidence-overlay work. The active sparse map used a
configured 1,024 MiB ceiling. Its reported block-pool allocation peaked at 895.97 MiB, below
that ceiling; the single short capture did not naturally require a rollover, while the
automated Open3D integration test forces travel rollover and verifies that the completed host
submap remains in the combined preview.

| Metric | P0 baseline | P2 bounded live engine |
| --- | ---: | ---: |
| Pose latency p95 | 46.25 ms | 62.05 ms |
| Point-map publication latency p95 | 718.70 ms | 330.80 ms |
| Mesh publication latency p95 | 1,214.25 ms | 1,049.45 ms |
| Accepted / integrated | 71 / 15 frames | 80 / 17 frames |
| Final provisional geometry | 59,263 points / 105,468 triangles | 59,274 points / 105,560 triangles |
| Tracking / mapping / journal drops | 10 / 0 / 0 | 1 / 0 / 0 |
| Peak worker working set | 923.33 MiB | 924.22 MiB |
| Peak allocated live map | not reported | 895.97 MiB |
| Coverage / tracking-overlay snapshots | not available | 10 / 17 |

P2 therefore preserves the provisional artifact, cuts the point-map latency tail by 54%, and
reduces tracker queue loss from ten frames to one without increasing mapping or journal loss.
Pose p95 regressed by 15.8 ms and remains an explicit P3 tracking/loop-correction target. The
mesh statistic has only two low-rate samples by design and should not be read as a stable
distribution. Coverage finished at 95.0% repeatedly observed, 1.7% weak, and 3.3% single-view.
As in P0, Windows `nvidia-smi` did not expose per-process VRAM, so the hard ceiling is validated
from the VoxelBlockGrid allocation contract and active-block telemetry rather than a driver
process counter.
