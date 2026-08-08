# Azure Kinect / Femto Mega capture worker

`rgbd-capture-worker.exe` is the Windows backend for Azure Kinect DK and Orbbec Femto Mega. It captures synchronized depth/RGB, performs calibrated color-to-depth alignment, rotates IMU samples into the depth camera, streams full-rate RGB-D, and archives schema-3 frames asynchronously. Femto Mega's distorted native depth raster is resampled through the Orbbec calibration XY table onto a wide-FOV virtual pinhole camera before streaming or archiving; the aligned color uses the identical remap. In Narrow mode, valid depth pixels that fall just beyond the tilted RGB camera's vertical boundary receive the nearest calibrated native-RGB boundary sample instead of an artificial black color.

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

RGB sensor controls use `--sensor-fps 0|5|15|25|30`, `--rgb-resolution auto|720p|1080p|1440p|1536p|2160p|3072p`, `--rgb-auto-exposure true|false`, `--rgb-exposure-us`, `--rgb-gain`, `--rgb-auto-white-balance true|false`, and `--rgb-white-balance-k`. Enable deterministic image processing with `--rgb-color-adjustments true` and pass brightness, contrast, saturation, and sharpness values; Azure additionally accepts backlight compensation. `--rgb-powerline-hz 50|60` selects anti-flicker. Azure's 3072p mode and its 2160p mode in ScanLan require a 5/15 fps sensor rate; 25 fps is Femto-only and 3072p is Azure-only. Unsupported resolution/profile combinations fail before capture. `--rgb-quality` and `--max-rgb-dimension` affect only the archived JPEG, not the sensor stream.

Femto IMU selection uses `--imu-accel-rate`, `--imu-accel-range`, `--imu-gyro-rate`, and `--imu-gyro-range`; zero selects the first device-default profile. Requested nonzero values must match an SDK profile exactly. Acceleration ranges are 2/4/8/16 g, gyro ranges are 125/250/500/1000/2000 degrees per second, and rates are expressed in Hz. Azure Kinect ignores these profile flags because its Sensor SDK does not expose IMU rate/range controls.
