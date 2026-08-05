# Azure Kinect / Femto Mega capture worker

`rgbd-capture-worker.exe` is the Windows backend for Azure Kinect DK and Orbbec Femto Mega. It captures synchronized native depth/RGB, performs calibrated color-to-depth alignment, rotates IMU samples into the depth camera, streams full-rate RGB-D, and archives schema-3 frames asynchronously.

## SDK discovery

- Azure Kinect: install Sensor SDK 1.4.x or set `AZURE_KINECT_SDK_ROOT`.
- Femto Mega: install Orbbec SDK v2.7.6, update the camera to the matching vendor-recommended firmware (1.3.1 at the time of this refactor), or set `ORBBEC_SDK_ROOT`.

CMake enables each SDK it finds. Missing SDKs do not prevent the executable from building; selecting an unavailable backend returns a clear error.

## Standalone checks

```powershell
rgbd-capture-worker.exe --capabilities
rgbd-capture-worker.exe --list
rgbd-capture-worker.exe --probe --sensor azure_kinect --connection usb
rgbd-capture-worker.exe --probe --sensor femto_mega --connection usb
rgbd-capture-worker.exe --probe --sensor femto_mega --connection network --address 192.168.1.10
```

`--capabilities` reports compiled backends without touching hardware. `--list` passively enumerates devices; Azure entries use their current SDK index so serial lookup is deferred until capture. Femto network addresses accept `IP` or `IP:PORT`; the default port is 8090.

## Capture

```powershell
rgbd-capture-worker.exe --phase C:\Scans\Room\phases\take-1 --id take-1 --name "Room take 1" --sensor femto_mega --connection usb --depth-fov narrow --fps 10 --max-depth 4.5 --imu --stream-rgbd
```

Add `--depth-binned` for 2×2 depth sampling. `--fps` controls the archive only; the binary `SCANRGBD` stream remains at sensor rate. Create `stop.flag` in the phase directory to flush and stop.
