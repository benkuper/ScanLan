# Scan archive format — schema 3

Schema 3 is the only supported project and phase format. ScanLan does not migrate older projects.

```text
scan-project/
├── project.json
├── supplemental-photos.json       # optional localized high-resolution texture views
├── supplemental/                  # optional orientation-normalized lossless photos
├── phases/
│   └── <phase-id>/
│       ├── phase.json
│       ├── frames.csv
│       ├── tracking.jsonl
│       ├── imu.csv                 # when enabled
│       ├── sensor.log
│       ├── depth/000000.u16
│       ├── color/000000.rgb
│       └── rgb/000000.jpg
└── outputs/
    ├── jobs/<job-id>.json
    ├── cache/local-phases/*.npz
    ├── cache/datasets/<fingerprint>/
    ├── room-cloud.ply
    ├── room-mesh.obj
    ├── room-mesh.mtl
    ├── room-texture.png
    ├── room-splat.ply
    ├── room-splat.preview.splat
    ├── room-splat.transform.json
    ├── splat-manifest.json
    ├── camera-poses.json
    ├── photo-localization-progress.json
    ├── preview.json
    ├── result.json
    └── progress.json
```

## Project manifest

`project.json` contains the project identity, capture settings, phase summaries, independent artifact records, current job, and reconstruction diagnostics. Required capture settings are:

```json
{
  "schemaVersion": 3,
  "settings": {
    "captureFps": 10,
    "maxDepthM": 4.2,
    "voxelSizeMm": 10,
    "sensorKind": "femto_mega",
    "sensorId": "femto_mega:usb:serial",
    "sensorConnection": "usb",
    "sensorAddress": "",
    "useImu": true,
    "depthFieldOfView": "narrow",
    "depthBinned": false,
    "rgbJpegQuality": 92,
    "maxRgbDimension": 0,
    "liveReconstruction": "points"
  },
  "artifacts": {
    "pointCloud": null,
    "texturedMesh": null,
    "gaussianSplat": null
  },
  "activeJob": null
}
```

An artifact records its relative path, status, source fingerprint, update time, metric flag, and staleness. Rebuilding one representation does not remove another.

## Supplemental photo manifest

`supplemental-photos.json` is optional schema 1. `photos` contains only accepted views used by texture baking. `attempts` is the persistent UI registry and also retains queued, currently localizing, and rejected files with their validation reason. Every accepted photo records its project-relative lossless image path, original source path, pinhole intrinsics, row-major `worldFromCamera`, depth-backed match and inlier counts, final reprojection RMSE, and a 0–100 quality score. Camera coordinates use OpenCV axes and world units remain metres. The reconstruction worker rejects missing images, non-finite poses, and unknown manifest versions.

```json
{
  "schemaVersion": 1,
  "coordinateConvention": "scanlan_world_opencv_camera_axes",
  "photos": [{
    "id": "content-hash",
    "path": "supplemental/content-hash.png",
    "camera": { "width": 4032, "height": 3024, "fx": 3100.0, "fy": 3100.0, "cx": 2015.5, "cy": 1511.5, "model": "pinhole", "distortion": [] },
    "worldFromCamera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "inlierCount": 184,
    "reprojectionRmsePixels": 1.21,
    "qualityScore": 88,
    "qualityLabel": "Excellent"
  }],
  "attempts": [{
    "id": "content-hash",
    "name": "view-01",
    "status": "localized",
    "qualityScore": 88
  }]
}
```

`outputs/photo-localization-progress.json` is an atomic schema-1 checkpoint with batch status, stage, detail, overall progress, and accepted/rejected counters. The desktop UI polls it so a reloaded webview can reconnect to an active worker.

## Phase manifest

Every `phase.json` requires:

- schema version, identity, timestamps, frame count, and duration;
- sensor identity and connection;
- depth intrinsics, scale, reliable range, FOV, and binning;
- calibrated RGB intrinsics, distortion coefficients, and dimensions for that sensor’s archived RGB source;
- a row-major 4×4 `rgbFromDepth` calibration;
- calibrated-RGB storage format, quality, resolution policy, and drop count;
- optional calibrated IMU metadata.

Depth and RGB timestamps are device microseconds. All metric transforms use metres.
RGB calibration is fail-closed: `pinhole` carries no coefficients,
`brown_conrady` carries `[k1,k2,p1,p2,k3]`, and `opencv_rational` carries
`[k1,k2,p1,p2,k3,k4,k5,k6]`. Azure Kinect and supported Femto Mega
profiles use the rational form; unsupported lens models stop capture instead
of being approximated.

## Frame index

`frames.csv` has this fixed header:

```text
index,source_sequence,timestamp_us,depth_path,color_path,rgb_path,rgb_timestamp_us,m00,...,m33
```

- `index` is dense archive order.
- `source_sequence` is the full-rate sensor sequence and may contain gaps by design.
- `depth_path` and `color_path` are always present.
- `rgb_path` is empty for Kinect and can also be empty if the bounded modern-camera JPEG queue dropped that native image. Consumers then use the aligned RGB frame with depth-camera intrinsics and an identity depth-to-RGB transform.
- `m00…m33` contains an optional row-major camera-to-world pose, supplied by Kinect Fusion.

`depth/*.u16` is little-endian `width × height` unsigned depth in millimetres. Zero is invalid. `color/*.rgb` is `width × height × 3` RGB8 already aligned to the depth camera. Femto Mega frames are distortion-corrected through the Orbbec calibration XY table onto a wide-FOV virtual pinhole grid; the manifest `camera` intrinsics describe that grid, and aligned color receives the same remap. Narrow depth extends slightly beyond the tilted RGB sensor at its vertical boundary, so those otherwise valid edge samples use the nearest calibrated native-RGB boundary color rather than a synthetic black hole. On Azure/Femto, `rgb/*.jpg` is the synchronized native RGB-camera view preferred for final texturing and 2DGS training. Kinect uses the aligned RGB source exclusively because its SDK coordinate mapper is the authoritative calibration.

## Tracking journal

`tracking.jsonl` is schema 1, one compact JSON record per processed full-rate frame:

```json
{
  "schemaVersion": 1,
  "sequence": 127,
  "depthTimestampUs": 42318421,
  "state": "tracking",
  "accepted": true,
  "integrated": false,
  "reason": "accepted",
  "overlap": 0.68,
  "inlierRatio": 0.73,
  "depthRmseMm": 11.2,
  "worldToCamera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
}
```

The production loader matches the journal to archived frames by `source_sequence`. Explicitly rejected frames are excluded. A complete accepted trajectory is inverted to camera-to-world and validated before being used as the final-pass seed.

`live_loops.jsonl` is schema 1, one record per bounded nonlocal submap query. Each record stores
the source and target submap IDs, sensor sequence, accepted state, ICP fitness/RMSE and
correspondence count, row-major target-from-source transform, 6 x 6 information matrix,
pose-graph safety result, and `requiresProductionRevalidation: true`. An accepted live record
is not permission for production fusion; the offline solver independently verifies the raw
observations and may reject it.

## Provisional live-session artifact

Stopping a capture atomically publishes the latest validated live map beneath `outputs/live/`:

```text
outputs/live/
|-- session.json
|-- poses.jsonl
|-- loops.jsonl
|-- tracking-summary.json
|-- latest-preview.bin
|-- latest-preview.ply
|-- latest-preview.glb
|-- submaps/
`-- coverage/
```

`session.json` schema 1 uses live contract 2. It identifies the source phase, calibration,
sensor, scale status, tracking counts, queue losses, memory telemetry, loop decisions, preview
paths, and deterministic final-map fingerprint. The PLY and GLB are explicitly provisional
point representations; production outputs never overwrite them. `latest-preview.bin` retains
the bounded internal `K2P1` packet so the desktop can reopen the exact final live view without
converting the raw capture. `poses.jsonl` is a snapshot of the fail-closed tracking journal.

Live contract 2 defines these tracking states: `ready`, `preview`, `tracking`, `searching`,
`relocalized`, `frozen`, `failed`, and `complete`. It also reserves engine message kind 5 for
coverage summaries and kind 6 for submap descriptors. Both are UTF-8 JSON with
`contractVersion: 2`; geometry messages remain the bounded binary point/mesh packets.

## IMU

`imu.csv` fields are:

```text
timestamp_us,type,x,y,z,temperature_c
```

Acceleration is `m/s²`; angular velocity is `rad/s`. Native workers rotate both into the calibrated depth-camera frame. Gyro samples are integrated between RGB-D timestamps to provide an odometry rotation prior, never an unconstrained final pose.

## Coordinate conventions

- right-handed metric world;
- OpenCV camera axes: +X right, +Y down, +Z forward;
- row-major transforms on disk;
- internal pose artifacts are camera-to-world unless their field explicitly says `worldToCamera`;
- viewer packets apply display-axis conversion separately and never rewrite the reconstruction.

## Canonical posed dataset

The production pass writes a fingerprinted schema-3 dataset beneath
`outputs/cache/datasets`. Each selected keyframe contains RGB and metric depth
projected directly onto the same bounded-resolution pinhole grid, a robust
depth mask, `depthConfidence`, `generatedDepthMask`, `depthProvenance`, and
`worldFromRgbCamera`. Confidence is 255 for valid measured depth and lower for
quality-gated generated depth; the provenance mask is 255 only where LingBot
filled an original sensor hole. Native RGB is undistorted while it is
resampled, rather than producing an intermediate 2K/4K depth map. 2DGS rejects
distorted inputs because its rasterizer traces pinhole rays. There is no COLMAP
or arbitrary-scale media path.
