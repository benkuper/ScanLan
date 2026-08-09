$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot "build"
$Open3DSource = Join-Path $BuildRoot "open3d-cuda-src"
$WheelRoot = Join-Path $BuildRoot "open3d-cuda-wheel"
$WorkerPython = Join-Path $BuildRoot "worker-venv/Scripts/python.exe"
$Open3DCpuBuild = Join-Path $BuildRoot "open3d-cpu-companion-build"

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
  throw "cmake was not found. Install Visual Studio C++ tools before building CUDA Open3D."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git was not found. Install Git before building CUDA Open3D."
}

# Prefer a compatible CUDA toolkit from the default installation root. CUDA's installer
# updates PATH for new terminals only, so relying exclusively on Get-Command
# makes a just-installed toolkit look unavailable to an existing Codex session.
$NvccCandidates = @()
$CudaInstallRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit/CUDA"
if (Test-Path $CudaInstallRoot) {
  $NvccCandidates += Get-ChildItem -Path $CudaInstallRoot -Directory |
    Where-Object {
      $match = [regex]::Match($_.Name, '^v(\d+)\.(\d+)$')
      $match.Success -and (
        [int]$match.Groups[1].Value -ge 13 -or
        ([int]$match.Groups[1].Value -eq 12 -and [int]$match.Groups[2].Value -ge 8)
      )
    } |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName "bin/nvcc.exe" }
}
if ($env:CUDA_PATH) {
  $NvccCandidates += Join-Path $env:CUDA_PATH "bin/nvcc.exe"
}
$NvccOnPath = Get-Command nvcc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if ($NvccOnPath) {
  $NvccCandidates += $NvccOnPath
}
$NvccCommand = $NvccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $NvccCommand) {
  throw "CUDA Toolkit 12.8+ or 13.x was not found. Install a Blackwell-capable CUDA toolkit, then retry."
}

$NvccVersionText = (& $NvccCommand --version) -join "`n"
$VersionMatch = [regex]::Match($NvccVersionText, 'release\s+(\d+)\.(\d+)')
if (-not $VersionMatch.Success) {
  throw "Could not determine the installed CUDA Toolkit version from nvcc."
}
$CudaVersion = [Version]::new(
  [int]$VersionMatch.Groups[1].Value,
  [int]$VersionMatch.Groups[2].Value
)
$CudaVersionSupported = $CudaVersion.Major -ge 13 -or (
  $CudaVersion.Major -eq 12 -and $CudaVersion.Minor -ge 8
)
if (-not $CudaVersionSupported) {
  throw "Open3D on the RTX 5080 requires CUDA 12.8 or newer (found $CudaVersion at $NvccCommand)."
}
$CudaToolkitRoot = Split-Path -Parent (Split-Path -Parent $NvccCommand)
$Open3DBuild = Join-Path $BuildRoot "open3d-cuda-build-$CudaVersion"
$env:CUDA_PATH = $CudaToolkitRoot
$CudaRuntimeBin = Join-Path $CudaToolkitRoot "bin/x64"
$env:PATH = $CudaRuntimeBin + [IO.Path]::PathSeparator + (Join-Path $CudaToolkitRoot "bin") + [IO.Path]::PathSeparator + $env:PATH
# CUDA's supported Visual Studio list can lag the newest MSVC toolset. This is
# nvcc's compatibility escape hatch for a newer host compiler.
if (-not $env:NVCC_PREPEND_FLAGS) {
  $env:NVCC_PREPEND_FLAGS = "--allow-unsupported-compiler"
}
if (-not (Test-Path $WorkerPython)) {
  throw "The worker Python environment is missing. Run npm run prepare:runtime once, then retry."
}

# Open3D's pip-package target imports wheel from the selected Python environment.
# Open3D 0.19 pins wheel 0.38.4, whose bdist_wheel module still imports
# pkg_resources. Setuptools removed pkg_resources in 82.0.0, so pin the final
# compatible release as part of the reproducible packaging toolchain.
& $WorkerPython -m pip install --disable-pip-version-check `
  "setuptools==81.0.0" `
  "wheel==0.38.4"
if ($LASTEXITCODE -ne 0) { throw "Installing Open3D's wheel build prerequisite failed." }
& $WorkerPython -c "import pkg_resources; from wheel.bdist_wheel import bdist_wheel"
if ($LASTEXITCODE -ne 0) { throw "Open3D's pinned Python packaging toolchain is incompatible." }

New-Item -ItemType Directory -Force -Path $BuildRoot, $WheelRoot | Out-Null
if (-not (Test-Path (Join-Path $Open3DSource ".git"))) {
  if (Test-Path $Open3DSource) {
    throw "$Open3DSource exists but is not an Open3D git checkout. Move it aside and retry."
  }
  & git clone --branch v0.19.0 --depth 1 https://github.com/isl-org/Open3D.git $Open3DSource
  if ($LASTEXITCODE -ne 0) { throw "Open3D source checkout failed." }
}

$CompatibilityPatch = Join-Path $PSScriptRoot "patches/open3d-0.19-cuda13-cmake4.patch"
$StdgpuCompatibilityPatch = (Resolve-Path (Join-Path $PSScriptRoot "patches/stdgpu-cuda13.patch")).Path -replace '\\', '/'
$PreviousErrorActionPreference = $ErrorActionPreference
try {
  # A failed reverse check is expected for a fresh checkout. Windows PowerShell
  # otherwise promotes git's stderr to a terminating NativeCommandError before
  # the exit code can select the normal patch-application path below.
  $ErrorActionPreference = "SilentlyContinue"
  & git -C $Open3DSource apply --reverse --check $CompatibilityPatch 2>$null
  $PatchAlreadyApplied = $LASTEXITCODE -eq 0
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
if (-not $PatchAlreadyApplied) {
  & git -C $Open3DSource apply --check $CompatibilityPatch
  if ($LASTEXITCODE -ne 0) {
    throw "The Open3D CUDA compatibility patch does not apply cleanly. Recreate $Open3DSource and retry."
  }
  & git -C $Open3DSource apply $CompatibilityPatch
  if ($LASTEXITCODE -ne 0) { throw "Open3D CUDA compatibility patch failed." }
}

# A full Open3D CUDA wheel contains both pybind modules: CUDA for accelerated
# tensor pipelines and CPU as its import-time fallback. Build the CPU companion
# once in a separate cache so switching configurations cannot invalidate the
# expensive CUDA objects, then place it beside the CUDA module for packaging.
$CpuPybindDestination = Join-Path $Open3DBuild "lib/Release/Python/cpu"
$CpuPybind = Get-ChildItem -Path $CpuPybindDestination -Filter "pybind*.pyd" -File -ErrorAction SilentlyContinue |
  Select-Object -First 1
if (-not $CpuPybind) {
  $CpuConfigureArguments = @(
    "-S", $Open3DSource,
    "-B", $Open3DCpuBuild,
    "-A", "x64",
    "-DBUILD_CUDA_MODULE=OFF",
    "-DBUILD_PYTHON_MODULE=ON",
    "-DSTATIC_WINDOWS_RUNTIME=OFF",
    "-DBUILD_GUI=OFF",
    "-DBUILD_WEBRTC=OFF",
    "-DBUILD_JUPYTER_EXTENSION=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_UNIT_TESTS=OFF",
    "-DBUILD_BENCHMARKS=OFF",
    "-DBUILD_PYTORCH_OPS=OFF",
    "-DBUILD_TENSORFLOW_OPS=OFF",
    "-DBUNDLE_OPEN3D_ML=OFF",
    "-DBUILD_AZURE_KINECT=OFF",
    "-DBUILD_LIBREALSENSE=OFF",
    "-DWITH_OPENMP=ON",
    "-DDEVELOPER_BUILD=OFF",
    "-DPython3_EXECUTABLE=$WorkerPython"
  )
  & $CMakeCommand @CpuConfigureArguments
  if ($LASTEXITCODE -ne 0) { throw "Open3D CPU companion configuration failed." }

  & $CMakeCommand --build $Open3DCpuBuild --config Release --target pybind --parallel
  if ($LASTEXITCODE -ne 0) { throw "Open3D CPU companion build failed." }

  $BuiltCpuPybind = Get-ChildItem -Path (Join-Path $Open3DCpuBuild "lib/Release/Python/cpu") -Filter "pybind*.pyd" -File |
    Select-Object -First 1
  if (-not $BuiltCpuPybind) { throw "Open3D CPU companion finished building, but its Python module was not found." }
  New-Item -ItemType Directory -Force -Path $CpuPybindDestination | Out-Null
  Copy-Item -Force -LiteralPath $BuiltCpuPybind.FullName -Destination $CpuPybindDestination
}

$ConfigureArguments = @(
  "-S", $Open3DSource,
  "-B", $Open3DBuild,
  "-A", "x64",
  "-T", "cuda=$CudaToolkitRoot",
  "-DBUILD_CUDA_MODULE=ON",
  "-DCMAKE_CUDA_ARCHITECTURES=120",
  "-DCMAKE_CUDA_COMPILER:FILEPATH=$NvccCommand",
  "-DCMAKE_CUDA_FLAGS:STRING=-Xcompiler=/Zc:preprocessor",
  "-DCUDAToolkit_ROOT=$CudaToolkitRoot",
  "-DSCANLAN_STDGPU_CUDA13_PATCH:FILEPATH=$StdgpuCompatibilityPatch",
  "-DBUILD_PYTHON_MODULE=ON",
  "-DSTATIC_WINDOWS_RUNTIME=OFF",
  "-DBUILD_GUI=OFF",
  "-DBUILD_WEBRTC=OFF",
  "-DBUILD_JUPYTER_EXTENSION=OFF",
  "-DBUILD_EXAMPLES=OFF",
  "-DBUILD_UNIT_TESTS=OFF",
  "-DBUILD_BENCHMARKS=OFF",
  "-DBUILD_PYTORCH_OPS=OFF",
  "-DBUILD_TENSORFLOW_OPS=OFF",
  "-DBUNDLE_OPEN3D_ML=OFF",
  "-DBUILD_AZURE_KINECT=OFF",
  "-DBUILD_LIBREALSENSE=OFF",
  "-DWITH_OPENMP=ON",
  "-DDEVELOPER_BUILD=OFF",
  "-DPython3_EXECUTABLE=$WorkerPython"
)
& $CMakeCommand @ConfigureArguments
if ($LASTEXITCODE -ne 0) { throw "CUDA Open3D configuration failed." }

& $CMakeCommand --build $Open3DBuild --config Release --target pip-package --parallel
if ($LASTEXITCODE -ne 0) { throw "CUDA Open3D build failed." }

$Wheel = Get-ChildItem -Path $Open3DBuild -Recurse -Filter "open3d*.whl" -File |
  Sort-Object LastWriteTimeUtc -Descending |
  Select-Object -First 1
if (-not $Wheel) {
  throw "Open3D finished building, but no Python wheel was found under $Open3DBuild."
}
$InstalledWheel = Join-Path $WheelRoot $Wheel.Name
Copy-Item -Force -LiteralPath $Wheel.FullName -Destination $InstalledWheel
Write-Host "CUDA Open3D wheel: $InstalledWheel"

& npm.cmd run prepare:runtime
if ($LASTEXITCODE -ne 0) { throw "The CUDA reconstruction worker failed to package." }
Write-Host "CUDA reconstruction worker is ready. npm run debug and npm run release will select it automatically."
