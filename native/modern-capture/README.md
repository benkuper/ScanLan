# Azure Kinect and Orbbec capture worker

`rgbd-capture-worker` records sensor-neutral, color-aligned RGB-D phases from Azure Kinect DK and Orbbec Femto Mega. Femto Mega supports both USB and network connections. Both backends can record calibrated accelerometer and gyroscope samples in the depth-camera coordinate frame.

## SDK discovery

- Azure Kinect: set `AZURE_KINECT_SDK_ROOT`, or install SDK 1.4.x in its standard Program Files location.
- Orbbec: set `ORBBEC_SDK_ROOT`, or install Orbbec SDK v2 under a versioned `C:\Program Files\OrbbecSDK*` directory.

The worker always builds. A backend whose SDK was not found reports a clear runtime error, allowing Orbbec-only or Azure-only application builds. SDK DLLs and Orbbec extension directories are copied beside the executable.

## Standalone checks

```powershell
.\rgbd-capture-worker.exe --list
.\rgbd-capture-worker.exe --probe --sensor azure_kinect --connection usb
.\rgbd-capture-worker.exe --probe --sensor femto_mega --connection usb
.\rgbd-capture-worker.exe --probe --sensor femto_mega --connection network --address 192.168.1.10
```

`--list` prints one JSON entry per connected Azure Kinect or Femto Mega, using a stable serial-based device ID where available. Pass that ID with `--device` to select a specific unit when more than one is attached.

Capture with IMU:

```powershell
.\rgbd-capture-worker.exe --phase C:\Scans\phase-id --id phase-id --name Room --sensor femto_mega --connection usb --depth-fov wide --depth-binned --fps 10 --max-depth 8 --imu
```

`--depth-fov` accepts `narrow` (the default) or `wide`. Add `--depth-binned` for 2×2 binning; omit it for unbinned depth. These options apply to both Azure Kinect and Femto Mega.

Create `stop.flag` in the phase directory to end capture cleanly.
