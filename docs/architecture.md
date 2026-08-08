# Runtime architecture

ScanLan separates realtime latency from archival throughput and final quality. Every boundary is explicit and versioned.

## Processes

| Process | Owns | Must not own |
|---|---|---|
| Tauri application | lifecycle, project state, status snapshots, exports | camera SDK state or reconstruction kernels |
| Native capture worker | camera SDK, calibration, synchronized RGB-D, IMU conversion | UI rendering or Open3D |
| Reconstruction worker | tracking, relocalization, TSDF, pose graph, point/mesh build | camera SDK |
| Splat worker | Photo/video decoding, COLMAP camera solving, CUDA 2DGS/3DGS optimization, checkpoints | RGB-D capture and TSDF reconstruction |

The realtime engine is started and reports `ready` before the camera is opened. Tauri pipes camera stdout directly into engine stdin and drains engine stdout on a dedicated thread. Sensor stderr goes to `sensor.log`, so a full pipe cannot stall capture.

## Live data path

`SCANRGBD` version 1 frames carry:

- sequence and device timestamps;
- pinhole depth-grid dimensions and intrinsics (Femto Mega native depth is calibrated and rectified before transport);
- depth scale and valid range;
- unsigned 16-bit depth and aligned RGB8;
- optional gyro delta quaternion;
- optional calibrated camera pose;
- the Kinect X-mirror flag.

The packed header is 164 bytes. Readers reject unknown versions, impossible calibration, oversized images, and payload-length mismatches before allocating geometry.

The capture-side stream queue has capacity 3 and discards its oldest unpublished item on overload. The Python reader feeds a latest-frame queue of 4; accepted keyframes feed a mapping queue of 8. None of these queues can grow without bound.

## Engine data path

`SCANENG1` version 1 multiplexes three message kinds:

- JSON status;
- packed point snapshots (`K2P1`, maximum 150,000 preview points);
- packed indexed meshes (`K2M2`, maximum 150,000 preview triangles).

Tauri validates message sizes and stores only the newest point and mesh packets in memory. UI polling returns a packet only when its frame sequence is newer than the caller’s. Reconstruction geometry is never polled from a growing file.

## Archive path

Archival depth/aligned-color writes and native-RGB JPEG compression happen behind bounded queues. The camera loop moves frame buffers into the queue and immediately returns to acquisition. If storage cannot keep up, the oldest pending archive frame is discarded and the drop is persisted in phase metadata.

Modern-camera RGB controls are applied before video streaming starts. Exposure is represented in microseconds across backends (Femto Mega converts to its 100 microsecond property units), while white balance is Kelvin. Orbbec integer properties are clamped and snapped to the range/step reported by the connected firmware and adjustments are logged. Manual Femto IMU requests select an exact advertised stream profile; a missing profile is a startup error rather than a silent fallback. The requested RGB and IMU configuration is persisted in `phase.json` for capture provenance.

`live.json` is a tiny, atomically replaced sensor heartbeat. During capture the UI reads its monotonic `frameCount`; it does not rescan `frames.csv`. The complete CSV is counted only during recovery or abnormal termination.

## Tracking state machine

The tracker persists across frames. A finite transform alone is insufficient for acceptance. A candidate must pass:

- metric depth correspondence count;
- overlap and inlier-ratio thresholds;
- depth RMSE;
- translation and angular-velocity limits.

On failure, the map freezes and the tracker searches a rotating bank spanning the complete accepted capture. Local continuation remains subject to strict continuity and IMU limits; a strong saved-keyframe match may instead relocalize anywhere in the known map after three consistent observations. The lock frame is not fused, and mapping resumes only on the next independently validated frame. Tracking acceptance and irreversible fusion use separate gates: a marginal pose may preserve local tracking continuity, but only high-overlap, high-inlier, low-residual keyframes reach the mapping thread.

Every decision is appended asynchronously to `tracking.jsonl`; no image data is duplicated. Raw RGB-D archival is intentionally independent from live pose acceptance: a rejected frame remains recoverable evidence, but it never enters the live TSDF map. The UI therefore reports raw archived, tracked, rejected, and fused-keyframe counts separately. The final loader matches entries by sensor sequence, excludes explicit rejections from the live pose seed, and can recover archived frames during offline optimization. The archive replay command deliberately includes rejected frames and omits derived journal poses so tracker changes remain testable.

## Shutdown

Normal stop creates `stop.flag`, lets the capture worker flush its archive, drains the mapper for a final geometry snapshot, closes the engine, and then publishes the completed phase manifest. Timeouts are bounded; a stuck child is terminated rather than leaving the UI indefinitely busy.

Unexpected phases are recovered from their manifest and CSV at the next launch. Derived jobs have independent checkpoints and can be cancelled without deleting raw captures.

Every active capture owns its sensor process, stdout relay, and realtime engine as one supervised unit. Dropping that unit after any startup, storage, or state error terminates all three, so a failed command cannot leave the camera or GPU worker running in the background.

Project and preference manifests are serialized to uniquely named sibling files, flushed to storage, and atomically replaced. Concurrent status or settings updates therefore cannot collide on one shared temporary path, and a failed publication leaves the previous valid manifest intact.
