# Kinect v2 capture worker

This small Windows-only executable keeps the legacy Kinect SDK outside the Tauri process. It records synchronized depth and color frames, maps color into depth-camera coordinates, and writes the phase format consumed by `worker/scanlan`.

## Build

Install Kinect for Windows SDK 2.0 and Visual Studio 2022 with C++ desktop support, then run from a Developer PowerShell:

```powershell
cmake -S native/kinect-capture -B build/kinect-capture -A x64
cmake --build build/kinect-capture --config Release
```

If the SDK installer did not define `KINECTSDK20_DIR`, pass `-DKINECT_SDK_ROOT="C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409"`.

The application build bundles the resulting executable and discovers it automatically. No worker-path environment variable is needed.

## Connection check

```powershell
legacy-capture-worker.exe --probe
```

The command succeeds only after the sensor opens and delivers synchronized depth and color frames.

## Standalone use

```powershell
legacy-capture-worker.exe --phase C:\Scans\project\phases\phase-id --id phase-id --name "North wall" --fps 10 --max-depth 4.2
```

Create an empty `stop.flag` inside the phase directory to end capture cleanly.
