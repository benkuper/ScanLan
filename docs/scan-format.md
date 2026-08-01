# ScanLan archive format

The project format is directory-based so interrupted captures remain recoverable and reconstruction can be rerun without the original sensor.

```text
scan-project/
├── project.json
├── phases/
│   └── <phase-id>/
│       ├── phase.json
│       ├── frames.csv
│       ├── depth/000000.u16
│       └── color/000000.rgb
└── outputs/
    ├── room-cloud.ply
    ├── room-mesh.obj
    ├── room-mesh.mtl
    ├── room-texture.png
    ├── camera-poses.json
    ├── preview.json
    └── result.json
```

## Depth frames

`*.u16` contains `width × height` unsigned 16-bit little-endian samples in row-major order. Values are millimetres. Zero means invalid or outside the configured reliable range.

The camera manifest also records `depth_field_of_view` (`narrow` or `wide`) and `depth_binned` so the selected Azure Kinect/Femto Mega depth mode can be reproduced.

## Color frames

`*.rgb` contains `width × height × 3` bytes in row-major RGB order. Color has already been sampled into depth-camera pixel coordinates, so color and depth have identical dimensions.

## Frame index

`frames.csv` always starts with:

```text
index,timestamp_us,depth_path,color_path,m00,...,m33
```

The optional 4×4 row-major matrix maps camera coordinates into the phase coordinate system. Mock and imported tracked data may supply it. Kinect captures leave it empty so Open3D estimates the trajectory offline.

## Sensor and IMU metadata

Schema-version 2 phase manifests can include a sensor descriptor and calibrated IMU stream:

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

`imu.csv` contains `timestamp_us,type,x,y,z,temperature_c`. `type` is `accel` or `gyro`. Capture workers rotate both vectors into the calibrated depth-camera frame. Reconstruction integrates gyro samples between RGB-D timestamps as an odometry rotation prior and falls back to identity-initialized RGB-D odometry when coverage is incomplete or the aided solve fails.

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

The matrix maps points in the current phase into the previous phase. A later UI milestone will create this file from three user-selected correspondences.

## Coordinate and unit conventions

- depth and exported point coordinates use metres
- camera coordinates are right-handed: +X right, +Y up, +Z forward
- PLY vertex colors are 8-bit sRGB
- an optional Unity export step will transform to Unity's coordinate convention

## Textured mesh and camera trajectory

`room-mesh.obj` is a triangle mesh reconstructed from selected, posed depth keyframes. Its UVs reproject each surface patch into the matching aligned RGB frame; `room-texture.png` is an atlas of those source images and `room-mesh.mtl` connects it to the OBJ. This preserves image detail independently of the point cloud's vertex colors.

`camera-poses.json` stores the reconstructed camera-to-viewer matrix, field of view, phase, timestamp, and source-frame index for every retained keyframe. `textureFrame` marks the subset used by the texture atlas. Matrices are flattened row-major 4×4 transforms.
