# Kinect v2 capture worker

`kinect2-capture-worker.exe` is the Windows-only Kinect v2 backend. It owns Kinect SDK 2.0 and Kinect Fusion outside the Tauri process, captures synchronized depth/color, uses the SDK coordinate mapper to align color exactly into depth geometry, streams full-rate RGB-D with validated Fusion poses, and archives schema-3 frames asynchronously.

## Build

Install Kinect for Windows SDK 2.0 and Visual Studio 2022 C++ desktop tools:

```powershell
cmake -S native/kinect-capture -B build/kinect-capture -A x64
cmake --build build/kinect-capture --config Release
```

If needed, pass `-DKINECT_SDK_ROOT="C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409"`.

Without the SDK, CMake builds an unavailable-backend stub so the rest of ScanLan still packages cleanly.

## Check and capture

```powershell
kinect2-capture-worker.exe --capabilities
kinect2-capture-worker.exe --probe
kinect2-capture-worker.exe --phase C:\Scans\Room\phases\take-1 --id take-1 --name "Room take 1" --fps 10 --max-depth 4.2 --stream-rgbd
```

`--capabilities` reports whether Kinect SDK support was compiled without opening the camera. The unavailable-backend stub returns an empty array.

Create `stop.flag` in the phase directory to flush and stop. With `--stream-rgbd`, stdout is binary `SCANRGBD` v1 and must be piped directly to the reconstruction engine; diagnostics use stderr.
