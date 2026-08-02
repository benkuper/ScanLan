# ScanLan archive format

The project format is directory-based so interrupted captures remain recoverable and reconstruction can be rerun without the original sensor.

```text
scan-project/
|-- project.json
|-- phases/
|   `-- <phase-id>/
|       |-- phase.json
|       |-- frames.csv
|       |-- depth/000000.u16
|       |-- color/000000.rgb
|       `-- rgb/000000.jpg
|-- sources/
|   `-- <source-id>/originals/
`-- outputs/
    |-- jobs/<job-id>.json
    |-- cache/datasets/<fingerprint>/
    |-- room-cloud.ply
    |-- room-mesh.obj
    |-- room-mesh.mtl
    |-- room-texture.png
    |-- room-splat.ply
    |-- room-splat.transform.json
    |-- splat-manifest.json
    |-- camera-poses.json
    |-- preview.json
    `-- result.json
```

## Project schema 2

Project schema 2 adds media sources, independent artifact records, and the durable active-job identifier. Legacy `outputPath`, `meshOutputPath`, and processing fields remain populated.

```json
{
  "schemaVersion": 2,
  "mediaSources": [],
  "artifacts": {
    "pointCloud": null,
    "texturedMesh": null,
    "gaussianSplat": null
  },
  "activeJob": null
}
```

Each artifact records its source fingerprint, metric/arbitrary scale, freshness, and last update. Rebuilding one artifact does not delete unrelated artifacts. Schema-1 projects migrate additively when opened.

## Depth and aligned color

`depth/*.u16` contains `width × height` unsigned 16-bit little-endian samples in row-major order. Values are millimetres. Zero means invalid or outside the configured reliable range.

`color/*.rgb` contains `width × height × 3` bytes in row-major RGB order. Color has already been sampled into depth-camera pixel coordinates, so color and depth have identical dimensions. This aligned RGB remains the authoritative input for odometry, TSDF fusion, and live preview.

The camera manifest also records `depth_field_of_view` (`narrow` or `wide`) and `depth_binned` so the selected Azure Kinect/Femto Mega depth mode can be reproduced.

## Native RGB and phase schema 3

Schema-version 3 captures additionally contain `rgb/*.jpg`. These are synchronized source-camera images at native resolution by default. JPEG encoding runs on a bounded background queue; an RGB queue drop does not invalidate the corresponding depth/aligned-RGB frame.

The phase manifest records the stored RGB camera model and the explicit transform from depth-camera coordinates into RGB-camera coordinates:

```json
{
  "schemaVersion": 3,
  "rgbCamera": {
    "width": 3840,
    "height": 2160,
    "fx": 1900.0,
    "fy": 1900.0,
    "cx": 1920.0,
    "cy": 1080.0,
    "model": "brown_conrady",
    "distortion": [0, 0, 0, 0, 0]
  },
  "rgbFromDepth": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  "sourceRgb": {
    "format": "jpeg",
    "quality": 92,
    "nativeResolution": true,
    "droppedFrames": 0
  }
}
```

`rgbFromDepth` is a row-major 4×4 transform. Old archives omit all three members and continue to use aligned `color_path` with identity calibration.

## Frame index

`frames.csv` starts with:

```text
index,timestamp_us,depth_path,color_path,rgb_path,rgb_timestamp_us,m00,...,m33
```

`rgb_path` and `rgb_timestamp_us` are optional. The optional 4×4 row-major matrix maps camera coordinates into the phase coordinate system. Mock and imported tracked data may supply it. Untracked captures leave it empty so Open3D estimates the trajectory offline.

## Sensor and IMU metadata

Phase manifests can include a sensor descriptor and calibrated IMU stream:

```json
{
  "poseSource": "imu_aided_offline",
  "sensor": {
    "kind": "femto_mega",
    "name": "Orbbec Femto Mega",
    "connection": "network",
    "serial": "...",
    "address": "192.168.1.10"
  },
  "imu": {
    "path": "imu.csv",
    "coordinateFrame": "depth_camera",
    "accelerationUnit": "m/s^2",
    "angularVelocityUnit": "rad/s"
  }
}
```

`imu.csv` contains `timestamp_us,type,x,y,z,temperature_c`. Capture workers rotate acceleration and angular velocity into the calibrated depth-camera frame. Reconstruction integrates gyro samples between RGB-D timestamps as an odometry rotation prior.

## Canonical posed-frame dataset

After final pose-graph optimization, ScanLan writes a fingerprinted dataset below `outputs/cache/datasets/`. `current.json` points to the current fingerprint. A frame contains native RGB, RGB-view metric depth, a depth-discontinuity mask, native intrinsics, and an unmodified `worldFromRgbCamera` matrix. RGB-view depth uses nearest-depth z-buffer collision resolution.

Canonical dataset convention:

- right-handed world
- metres for RGB-D captures; arbitrary scale for COLMAP media
- OpenCV camera axes: +X right, +Y down, +Z forward
- `worldFromCamera` transforms
- row-major matrices on disk

Viewer-oriented `camera-poses.json` is a separate display artifact and is never used as splat training input.

## Phase alignment

Automatic processing registers each phase against the preceding phase. If automatic global registration is ambiguous, add `manual_transform.json` to the later phase:

```json
{
  "toPrevious": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

The matrix maps points in the current phase into the previous phase.

## Point cloud, mesh, and camera trajectory

The artifact coordinate system is right-handed and uses metres for RGB-D captures. PLY vertex colors are 8-bit sRGB. Viewer exports use +Y up and record their display-axis conversion separately.

`room-mesh.obj` is reconstructed from selected posed depth keyframes. Native-RGB texturing transforms each sampled depth point through `rgbFromDepth`, applies RGB intrinsics/distortion, and rejects occluded samples against the same RGB-view z-buffer used by the canonical dataset. Atlas tiles crop unused pixels, extend padded edges, compensate exposure, and scale up to the configured 8K/16K capacity. Legacy archives follow the same path with aligned-RGB identity calibration.

`camera-poses.json` stores reconstructed camera-to-viewer matrices, field of view, phase, timestamp, and source-frame index. `textureFrame` marks the texture subset. These matrices are flattened row-major 4×4 transforms.

## Gaussian splats

`room-splat.ply` is a canonical 3D Gaussian Splatting PLY with:

```text
x y z
nx ny nz
f_dc_0..2
f_rest_0..44
opacity
scale_0..2
rot_0..3
```

`splat-manifest.json` records the source fingerprint, trainer/runtime versions, metric scale, loss configuration, and coordinate convention. `room-splat.transform.json` is applied at the Unity GameObject level. Non-uniform viewer transforms are not baked into Gaussian covariance or spherical harmonics.

RGB-D training initializes from the canonical metric cloud and combines RGB L1, SSIM, and masked robust expected-depth loss. Photo/video sources are durably copied, filtered, GPU-registered with COLMAP, and marked arbitrary-scale. The optional splat runtime uses CUDA mixed precision and gsplat's Splatfacto-style adaptive strategy, and remains isolated from the Open3D PyInstaller worker.

## Durable artifact jobs

`outputs/jobs/<job-id>.json` and its adjacent log record pipeline, targets, stage, progress, iteration/loss/ETA, fingerprint, status, and resumability. Only one Open3D CUDA/gsplat accelerator job runs at once. Cancellation requests create `outputs/cancel.flag`; splat checkpoints resume only when the source fingerprint still matches.
