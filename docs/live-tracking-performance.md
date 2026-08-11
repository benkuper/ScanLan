# Live tracking performance validation

The live-path optimization was validated on 11 August 2026 by replaying 300 consecutive retained
frames from the user's real 640x576 Orbbec Femto Mega room capture. The archive covers 19.963 s at
15 fps and retains the camera's original 30 fps source sequence. Both runs used the same Open3D
0.19 CUDA build, RTX 5080 Laptop GPU, 10 mm live voxel size, 1,024 MiB live-map budget, and points
mode. Replay was timestamp-paced.

| Measure | Previous path | Optimized path | Change |
| --- | ---: | ---: | ---: |
| Frames reaching the tracker | 181 / 300 | 275 / 300 | +52% |
| Tracking-queue drops | 119 | 25 | -79% |
| Accepted tracked frames | 167 | 261 | +56% |
| Rejected tracked frames | 14 | 14 | no increase |
| Pose latency median | 188.0 ms | 31.0 ms | 6.1x faster |
| Pose latency p95 | 791.7 ms | 291.3 ms | 2.7x faster |
| Point-preview latency median | 265.0 ms | 62.0 ms | 4.3x faster |
| Point-preview latency p95 | 1,306.6 ms | 481.8 ms | 2.7x faster |
| Point snapshots published | 13 | 25 | 1.9x as many |

The absolute rejection count remained 14 while the tracker evaluated 94 additional frames. The
accepted share of evaluated frames increased from 92.3% to 94.9%. The remaining tail spikes occur
during deliberate relocalization and submap completion rather than normal pose estimation.

The benchmark's `sourceDrops` counter is excluded here: archived frames preserve their original
even-numbered 30 fps sensor sequence, so timestamp-paced 15 fps replay intentionally appears to
have one source-sequence gap per retained frame. Live capture still reports actual transport gaps.

The change combines four independently bounded mechanisms:

- calibrated <=100k-pixel live odometry while retaining full-resolution archive/fusion input;
- explicit Open3D multi-scale iteration and convergence criteria;
- adjacent-frame fallback as soon as a source sequence gap is observed;
- incremental preview points instead of repeated full-TSDF extraction.

Gyroscope deltas are composed across both native and Python queue drops. Sustained tracking loss
holds retained recording after a three-frame evidence margin; recovery continues at sensor rate
and normal recording resumes only after a validated non-relocalization tracking pose.
