# ScanLan

ScanLan continuously previews a depth stream, records room-scale RGB-D captures in phases, and reconstructs them into a colored PLY point cloud plus an RGB-reprojected textured mesh. It supports Kinect v2, Azure Kinect DK, and Orbbec Femto Mega over USB or Ethernet.

## Current workflow

- Sensor discovery is manual: select **Capture profile → Available sensor → Scan sensors** when you are ready to connect. No hardware probes run in the background.
- The preferred physical device is saved by serial number but is not opened at launch. If it is absent, scan and choose another supported sensor; a Femto Mega can also be added directly by network IP.
- Kinect v2 is listed when its capture support is installed without probing or opening the camera. Its light and streams start only after Kinect v2 is explicitly selected.
- Live depth/color points remain visible even when a phase is not being recorded.
- Kinect v2 can use Kinect Fusion for live diagnostics. Azure Kinect and Femto Mega record calibrated IMU samples that seed offline RGB-D odometry.
- If live tracking is unavailable or lost, the original RGB-D frames remain usable by the offline Open3D fallback.
- Reconstruction reports its current stage, percent complete, point count, and estimated time remaining.
- The viewer supports point size, opacity, color display, manual translation/rotation, and click-assisted floor alignment.
- A chosen viewer pose can be applied to the exported PLY; the original is retained as `room-cloud.untransformed.ply`.
- Reconstructed keyframe cameras can be overlaid as phase-colored frustums and trajectory lines in the result viewer.
- Mesh export writes an OBJ/MTL/PNG bundle whose UVs reproject geometry into captured RGB images instead of reducing texture to vertex colors.
- Export PLY opens a native Save As window and writes a Unity-ready copy with its X axis corrected; the project PLY remains unchanged.

There is no demo point cloud and no browser/mock capture path in the production app.

## Architecture

```text
Svelte / Three.js UI
        | Tauri commands
Rust session manager -------- project.json + phase folders
        |                              |
        +-- legacy-capture-worker.exe       +-- scanlan-worker.exe
        |   Kinect v2 SDK + Fusion         Open3D / NumPy
        +-- rgbd-capture-worker.exe        IMU-aided odometry + TSDF
        |   Azure Kinect / Orbbec SDK
        |                              |
        +------------------------------+-- outputs/room-cloud.ply
                                       +-- outputs/room-mesh.obj + .mtl + texture.png
                                       +-- outputs/camera-poses.json
```

Each native worker maps color into the active depth-camera view and writes the same sensor-neutral archive. IMU measurements are rotated into depth-camera coordinates before being saved. The final offline pass estimates the trajectory, aligns separate phases, produces the cleaned point cloud, and builds a depth-connected surface whose UVs sample a packed atlas of posed RGB keyframes.

## Supported sensors

- **Kinect v2:** USB 3, Kinect for Windows SDK 2.0. This backend remains available but is optional when building for another sensor.
- **Azure Kinect DK:** USB 3, Azure Kinect Sensor SDK 1.4.x. Supports narrow/wide depth FOV with unbinned or 2×2-binned capture, color-to-depth calibration, and the accelerometer/gyro.
- **Orbbec Femto Mega:** USB or network, Orbbec SDK v2. Supports the same narrow/wide and unbinned/2×2-binned depth modes. The app automatically detects versioned installations under `C:\Program Files\OrbbecSDK*`; alternatively set `ORBBEC_SDK_ROOT`. Network mode accepts `IP` or `IP:PORT` and defaults to port 8090.

Choose a discovered device in **Capture profile → Available sensor**, or choose **Orbbec Femto Mega · Network IP…** and enter `IP` or `IP:PORT`. The choice persists across projects and restarts. For Azure Kinect and Femto Mega, leave **Use IMU to aid tracking** enabled unless diagnosing an IMU problem. IMU input supplies a rotation prior; RGB-D odometry is always retried without that prior if it cannot converge.

For Azure Kinect and Femto Mega, **Depth field of view** and **Depth binning** select NFOV/WFOV and unbinned/2×2-binned operation. The resulting modes are 640×576 at 30 fps, 320×288 at 30 fps, 1024×1024 at 15 fps, and 512×512 at 30 fps, respectively. **Maximum depth** can be set from 1.5 m through 8.0 m; samples beyond the selected limit are discarded.

## One-line debug workflow

Install the SDK for at least one supported camera, Visual Studio C++ desktop tools, stable Rust/MSVC, Node.js 20+, and Python 3.10-3.12. Then connect the selected sensor.

After the initial `npm install`, everyday iteration is one command:

```powershell
npm run debug
```

That command builds stale native/Python helpers, starts Tauri and Vite hot reload, bundles available camera runtimes beside their workers, and connects to the selected sensor. Running it a second time detects the active debug session instead of opening a duplicate.

To build and launch the optimized app without creating an installer:

```powershell
npm run release
```

The capture and Open3D reconstruction workers are already release-built in both workflows, so release mode primarily optimizes the Tauri shell and serves the production frontend bundle. It is useful for checking end-to-end responsiveness without debug and hot-reload overhead.

## Reconstruction acceleration

Reconstruction automatically uses a CUDA-enabled Open3D worker when one is bundled and falls back to the OpenMP CPU pipeline when CUDA is absent or an individual GPU operation is unavailable. The processing status shows the selected backend, and `outputs/result.json` records the backend, total processing time, and per-stage timings.

Unchanged phases are cached under each scan's `outputs/cache` directory. Rebuilding the same scan reuses validated camera tracking, selected keyframes, and local phase fusion. The cache is derived data, is fingerprinted from the phase inputs and reconstruction settings, and is invalidated automatically when those inputs change.

The stock Windows Open3D wheel is CPU-only. On an NVIDIA system, install CUDA Toolkit 12.8 or newer and then build the project-local CUDA worker:

```powershell
npm run prepare:runtime
npm run prepare:cuda
```

`prepare:cuda` builds Open3D 0.19 for the installed GPU architecture, places its wheel under `build/open3d-cuda-wheel`, and repackages `scanlan-worker.exe`. Subsequent `npm run debug` and `npm run release` commands discover that wheel automatically. Set `SCANLAN_DEVICE=cpu` before launching to force the CPU path while diagnosing a CUDA problem.

For the quickest CPU build, use 10–15 mm point spacing. With CUDA, settings below 10 mm use a streaming GPU voxel map so fine-cloud fusion remains accelerated without allocating a room-scale TSDF. The legacy CPU fallback is still substantially slower at fine spacing. One millimetre is supported for maximum detail, but produces much larger clouds and takes longer to process and export.

To create the Windows installer:

```powershell
npm run tauri -- build
```

## Capture guidance

- Watch the tracking status. If it changes to searching/lost, return to the last textured area until it locks again.
- Move slowly and keep useful geometry inside the configured reliable depth range.
- Start each additional phase while looking at textured geometry from an earlier phase.
- Prefer corners, furniture, rocks, or marker boards over blank walls and flat ground.
- Outdoors, scan at night, deep twilight, or in shade; direct sunlight can overwhelm infrared depth.
- One millimetre is the minimum supported reconstruction spacing. Settings below 10 mm use memory-bounded surfel merging instead of a room-scale TSDF.

## Verification

```powershell
npm run check
cargo check --manifest-path src-tauri/Cargo.toml
.\build\worker-venv\Scripts\python.exe -m unittest discover -s worker\tests -v
```

Project layout:

- `src/` - Svelte application
- `src-tauri/` - Rust state, storage, and worker orchestration
- `native/kinect-capture/` - Kinect SDK v2 and Kinect Fusion worker
- `native/modern-capture/` - Azure Kinect and Orbbec Femto Mega worker
- `worker/` - Open3D/NumPy reconstruction
- `docs/scan-format.md` - scan archive format

Licensed under the [GNU General Public License v3.0](LICENSE).
