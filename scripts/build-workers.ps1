$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot "build"
$KinectSource = Join-Path $ProjectRoot "native/kinect-capture"
$KinectBuild = Join-Path $BuildRoot "kinect-capture"
$ModernCaptureSource = Join-Path $ProjectRoot "native/modern-capture"
$ModernCaptureBuild = Join-Path $BuildRoot "modern-capture"
$MeshRepairSource = Join-Path $ProjectRoot "native/mesh-repair"
$MeshRepairBuild = Join-Path $BuildRoot "mesh-repair-cgal"
$WorkerVenv = Join-Path $BuildRoot "worker-venv"
$CudaWheelRoot = Join-Path $BuildRoot "open3d-cuda-wheel"
$WorkerBuildStamp = Join-Path $ProjectRoot "worker/dist/scanlan-worker.build.json"

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

function ConvertTo-ComparablePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  return ([IO.Path]::GetFullPath($Path) -replace '\\', '/').TrimEnd('/')
}

function Reset-StaleCMakeBuild {
  param(
    [Parameter(Mandatory = $true)][string]$SourceDirectory,
    [Parameter(Mandatory = $true)][string]$BuildDirectory
  )

  $CachePath = Join-Path $BuildDirectory "CMakeCache.txt"
  if (-not (Test-Path -LiteralPath $CachePath)) { return }

  $CachedSourceEntry = Get-Content -LiteralPath $CachePath |
    Where-Object { $_ -like "CMAKE_HOME_DIRECTORY:INTERNAL=*" } |
    Select-Object -First 1
  $CachedBuildEntry = Get-Content -LiteralPath $CachePath |
    Where-Object { $_ -like "CMAKE_CACHEFILE_DIR:INTERNAL=*" } |
    Select-Object -First 1
  $ExpectedSource = ConvertTo-ComparablePath $SourceDirectory
  $ExpectedBuild = ConvertTo-ComparablePath $BuildDirectory
  $CachedSource = if ($CachedSourceEntry) {
    ConvertTo-ComparablePath ($CachedSourceEntry -replace '^CMAKE_HOME_DIRECTORY:INTERNAL=', '')
  } else { $null }
  $CachedBuild = if ($CachedBuildEntry) {
    ConvertTo-ComparablePath ($CachedBuildEntry -replace '^CMAKE_CACHEFILE_DIR:INTERNAL=', '')
  } else { $null }

  if (($CachedSource -ine $ExpectedSource) -or ($CachedBuild -ine $ExpectedBuild)) {
    Write-Host "Removing stale CMake build state from $BuildDirectory"
    Remove-Item -LiteralPath $BuildDirectory -Recurse -Force
  }
}

Reset-StaleCMakeBuild -SourceDirectory $KinectSource -BuildDirectory $KinectBuild
Reset-StaleCMakeBuild -SourceDirectory $ModernCaptureSource -BuildDirectory $ModernCaptureBuild
Reset-StaleCMakeBuild -SourceDirectory $MeshRepairSource -BuildDirectory $MeshRepairBuild

$CMakeCommand = Get-Command cmake -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $CMakeCommand) {
  $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
  if (Test-Path $VsWhere) {
    $VisualStudioRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($VisualStudioRoot) {
      $BundledCMake = Join-Path $VisualStudioRoot "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
      if (Test-Path $BundledCMake) {
        $CMakeCommand = $BundledCMake
      }
    }
  }
}

if (-not $CMakeCommand) {
  throw "CMake was not found. Install Visual Studio with the Desktop development with C++ workload."
}

& $CMakeCommand -S $KinectSource -B $KinectBuild -A x64
if ($LASTEXITCODE -ne 0) { throw "Kinect capture worker configuration failed." }
& $CMakeCommand --build $KinectBuild --config Release
if ($LASTEXITCODE -ne 0) { throw "Kinect capture worker build failed." }
& $CMakeCommand -S $ModernCaptureSource -B $ModernCaptureBuild -A x64
if ($LASTEXITCODE -ne 0) { throw "Modern sensor worker configuration failed." }
& $CMakeCommand --build $ModernCaptureBuild --config Release
if ($LASTEXITCODE -ne 0) { throw "Modern sensor worker build failed." }

$VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { Join-Path $BuildRoot "vcpkg" }
$VcpkgToolchain = Join-Path $VcpkgRoot "scripts/buildsystems/vcpkg.cmake"
if (-not (Test-Path -LiteralPath $VcpkgToolchain)) {
  throw "vcpkg was not found. Set VCPKG_ROOT or clone vcpkg into build/vcpkg and run bootstrap-vcpkg.bat."
}
& $CMakeCommand -S $MeshRepairSource -B $MeshRepairBuild -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain" `
  "-DVCPKG_TARGET_TRIPLET=x64-windows"
if ($LASTEXITCODE -ne 0) { throw "CGAL mesh-repair worker configuration failed." }
& $CMakeCommand --build $MeshRepairBuild --config Release
if ($LASTEXITCODE -ne 0) { throw "CGAL mesh-repair worker build failed." }

$ReconstructionDirectory = Join-Path $ProjectRoot "worker/dist/scanlan-worker"
$ReconstructionExe = Join-Path $ReconstructionDirectory "scanlan-worker.exe"
$CudaWheel = Get-ChildItem -Path $CudaWheelRoot -Filter "open3d*.whl" -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTimeUtc -Descending |
  Select-Object -First 1
if ($CudaWheel) {
  $CudaRoots = @()
  if ($env:CUDA_PATH) { $CudaRoots += $env:CUDA_PATH }
  $CudaInstallRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit/CUDA"
  if (Test-Path $CudaInstallRoot) {
    $CudaRoots += Get-ChildItem -Path $CudaInstallRoot -Directory |
      Sort-Object Name -Descending |
      Select-Object -ExpandProperty FullName
  }
  $CudaRuntimeBin = $CudaRoots |
    ForEach-Object { @((Join-Path $_ "bin/x64"), (Join-Path $_ "bin")) } |
    Where-Object { @(Get-ChildItem -LiteralPath $_ -Filter "cudart64_*.dll" -File -ErrorAction SilentlyContinue).Count -gt 0 } |
    Select-Object -First 1
  if (-not $CudaRuntimeBin) {
    throw "A CUDA wheel is present, but its CUDA 12/13 runtime DLL directory was not found."
  }
  # CUDA toolkits place redistributable DLLs below bin or bin/x64. Put the
  # matching directory first so both CUDA 12.8 and CUDA 13 wheels package cleanly.
  # Open3D can load CUDA and PyInstaller can discover and embed dependencies.
  $env:PATH = $CudaRuntimeBin + [IO.Path]::PathSeparator + $env:PATH
}
$RuntimeKey = if ($CudaWheel) {
  "cuda:$($CudaWheel.Name):$((Get-FileHash -Algorithm SHA256 $CudaWheel.FullName).Hash)"
} else {
  "pypi:open3d>=0.19,<0.20"
}
$WorkerSources = @(
  Get-ChildItem -Path (Join-Path $ProjectRoot "worker/scanlan") -Recurse -File |
    Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' }
  Get-ChildItem -Path (Join-Path $ProjectRoot "validation/scanlan_validation") -Recurse -File |
    Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' }
  Get-ChildItem -Path (Join-Path $ProjectRoot "material/scanlan_material") -Recurse -File |
    Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' }
  Get-Item -Path (Join-Path $ProjectRoot "worker/entry.py")
  Get-Item -Path (Join-Path $ProjectRoot "worker/pyproject.toml")
  Get-Item -Path (Join-Path $ProjectRoot "worker/scanlan-worker.spec")
  Get-Item -Path (Join-Path $ProjectRoot "validation/pyproject.toml")
  Get-Item -Path (Join-Path $ProjectRoot "material/pyproject.toml")
  Get-Item -Path (Join-Path $ProjectRoot "scripts/build-workers.ps1")
)
$NewestWorkerSource = $WorkerSources | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
$NeedsWorkerBuild = -not (Test-Path $ReconstructionExe)
if (-not $NeedsWorkerBuild -and $NewestWorkerSource) {
  $NeedsWorkerBuild = $NewestWorkerSource.LastWriteTimeUtc -gt (Get-Item $ReconstructionExe).LastWriteTimeUtc
}
if (-not $NeedsWorkerBuild) {
  try {
    $PreviousBuild = Get-Content $WorkerBuildStamp -Raw | ConvertFrom-Json
    $NeedsWorkerBuild = $PreviousBuild.runtimeKey -ne $RuntimeKey
  } catch {
    $NeedsWorkerBuild = $true
  }
}

if ($NeedsWorkerBuild) {
  $PythonCommand = $null
  $PythonArgs = @()

  if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("3.11", "3.10", "3.12")) {
      & py -$version -c "import sys; print(sys.executable)" 2>$null | Out-Null
      if ($LASTEXITCODE -eq 0) {
        $PythonCommand = "py"
        $PythonArgs = @("-$version")
        break
      }
    }
  }

  if (-not $PythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $PythonCommand = "python"
  }

  if (-not $PythonCommand) {
    throw "Python 3.10-3.12 was not found. Install one of those versions to build the bundled reconstruction worker."
  }

  $WorkerPython = Join-Path $WorkerVenv "Scripts/python.exe"
  if (-not (Test-Path $WorkerPython)) {
    if ($PythonCommand -eq "py") {
      & py @PythonArgs -m venv $WorkerVenv
    } else {
      & $PythonCommand -m venv $WorkerVenv
    }
  }

  & $WorkerPython -m pip install --upgrade --force-reinstall --no-deps "$ProjectRoot/validation"
  if ($LASTEXITCODE -ne 0) { throw "Shared ScanLan validation engine could not be installed." }
  & $WorkerPython -m pip install --upgrade --force-reinstall --no-deps "$ProjectRoot/material"
  if ($LASTEXITCODE -ne 0) { throw "Shared ScanLan material foundation could not be installed." }

  if ($CudaWheel) {
    Write-Host "Using CUDA-enabled Open3D wheel: $($CudaWheel.FullName)"
    & $WorkerPython -m pip install -e "$ProjectRoot/worker[build]"
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction worker dependencies failed to install." }
    & $WorkerPython -m pip install --force-reinstall $CudaWheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled Open3D wheel failed to install." }
    & $WorkerPython -c "import open3d as o3d; assert o3d._build_config['BUILD_CUDA_MODULE'], 'Open3D wheel is not CUDA-enabled'; assert o3d.core.cuda.is_available(), 'Open3D CUDA backend cannot access a compatible device or runtime'; print(f'Open3D CUDA devices: {o3d.core.cuda.device_count()}')"
    if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled Open3D validation failed. Review the diagnostic above." }
  } else {
    & $WorkerPython -m pip install -e "$ProjectRoot/worker[reconstruction,build]"
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction worker dependencies failed to install." }
  }
  Push-Location (Join-Path $ProjectRoot "worker")
  try {
    # Open3D plus CUDA is close to 1 GB. A one-file executable unpacked that
    # payload on every preview/reconstruction launch and made the UI appear to
    # freeze. Keep the runtime extracted once and launch it directly instead.
    & $WorkerPython -m PyInstaller --noconfirm --clean --onedir --name scanlan-worker --collect-all open3d --collect-all scanlan_material --collect-all scanlan_validation entry.py
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction worker packaging failed." }
  } finally {
    Pop-Location
  }
  $Stamp = @{ runtimeKey = $RuntimeKey; builtAtUtc = [DateTime]::UtcNow.ToString("o") } | ConvertTo-Json
  [IO.File]::WriteAllText($WorkerBuildStamp, $Stamp + [Environment]::NewLine)
} else {
  Write-Host "Reconstruction worker is up to date."
}

$MeshRepairRuntime = Join-Path $MeshRepairBuild "Release"
$MeshRepairExe = Join-Path $MeshRepairRuntime "scanlan-mesh-repair.exe"
if (-not (Test-Path -LiteralPath $MeshRepairExe)) {
  throw "The CGAL mesh-repair executable was not produced."
}
Copy-Item -LiteralPath $MeshRepairExe -Destination $ReconstructionDirectory -Force
Get-ChildItem -LiteralPath $MeshRepairRuntime -Filter "*.dll" -File |
  Copy-Item -Destination $ReconstructionDirectory -Force

$KinectExe = Join-Path $KinectBuild "Release/kinect2-capture-worker.exe"
$ModernCaptureExe = Join-Path $ModernCaptureBuild "Release/rgbd-capture-worker.exe"
Write-Host "Kinect worker: $KinectExe"
Write-Host "Azure Kinect / Femto Mega worker: $ModernCaptureExe"
Write-Host "Reconstruction worker: $ReconstructionExe"
Write-Host "Mesh repair worker: $MeshRepairExe"
Write-Host "Reconstruction backend: $(if ($CudaWheel) { 'CUDA-capable Open3D' } else { 'CPU Open3D' })"
Write-Host "All workers will be discovered and bundled automatically."
