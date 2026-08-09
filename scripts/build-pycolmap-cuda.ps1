param(
  [ValidateSet("Native", "Redistributable")]
  [string]$Mode = "Native",
  [string]$ColmapVersion = "4.1.1",
  [string]$CudaArchitectures = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PackageRoot = Join-Path $ProjectRoot "splat-worker"
$RuntimeRoot = Join-Path $PackageRoot ".venv"
$Python = Join-Path $RuntimeRoot "Scripts/python.exe"
$RuntimeScripts = Join-Path $RuntimeRoot "Scripts"
$SitePackages = Join-Path $RuntimeRoot "Lib/site-packages"
$BuildRoot = Join-Path $ProjectRoot "build/pycolmap-cuda"
$SourceRoot = Join-Path $BuildRoot "colmap-$ColmapVersion"
$VcpkgRoot = Join-Path $BuildRoot "vcpkg"
$OnnxVersion = "1.28.0"
$OnnxArchiveName = "onnxruntime-win-x64-gpu_cuda13-$OnnxVersion.zip"
$OnnxArchive = Join-Path $BuildRoot $OnnxArchiveName
$OnnxArchiveSha256 = "137f0822a4923b1d84d3e09496e0792ebbb221eb3a61a0657f71a12ab68ab1e2"
$OnnxSourceRoot = Join-Path $BuildRoot "onnxruntime-win-x64-gpu_cuda13-$OnnxVersion"
$OnnxHeaderRoot = Join-Path $BuildRoot "onnxruntime-headers-$OnnxVersion"
$OnnxIncludeRoot = Join-Path $OnnxHeaderRoot "include"
$OnnxLibRoot = Join-Path $OnnxSourceRoot "lib"
$NativeBuildRoot = Join-Path $BuildRoot "native-$ColmapVersion-$($Mode.ToLowerInvariant())-onnx-$OnnxVersion"
$InstallRoot = Join-Path $NativeBuildRoot "install"
$VcpkgInstalledRoot = Join-Path $NativeBuildRoot "vcpkg_installed"
$RawWheelRoot = Join-Path $NativeBuildRoot "wheels-raw"
$RepairedWheelRoot = Join-Path $NativeBuildRoot "wheels-repaired"
$PythonBuildRoot = Join-Path $NativeBuildRoot "pycolmap-build"
$StampPath = Join-Path $SitePackages "pycolmap/scanlan-cuda-build.txt"
$VcpkgBaseline = "a0b1c8d3a477c1cb4813d8e127a56961707ca42b"
$Triplet = "x64-windows-release"

function Invoke-Checked([string]$Label, [scriptblock]$Command) {
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE."
  }
}

function Assert-ChildPath([string]$Parent, [string]$Child) {
  $ResolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
  $ResolvedChild = [System.IO.Path]::GetFullPath($Child)
  if (-not $ResolvedChild.StartsWith("$ResolvedParent$([System.IO.Path]::DirectorySeparatorChar)", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Build path escaped its expected root: $ResolvedChild"
  }
}

if (-not (Test-Path -LiteralPath $Python)) {
  throw "The splat Python runtime is missing. Create splat-worker/.venv before building CUDA PyCOLMAP."
}
if (-not $env:CUDA_PATH -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH "bin/nvcc.exe"))) {
  throw "A complete CUDA toolkit was not found through CUDA_PATH."
}

Assert-ChildPath $ProjectRoot $BuildRoot
Assert-ChildPath $BuildRoot $SourceRoot
Assert-ChildPath $BuildRoot $VcpkgRoot
Assert-ChildPath $BuildRoot $OnnxArchive
Assert-ChildPath $BuildRoot $OnnxSourceRoot
Assert-ChildPath $BuildRoot $OnnxHeaderRoot
Assert-ChildPath $BuildRoot $NativeBuildRoot
$null = New-Item -ItemType Directory -Force -Path $BuildRoot, $NativeBuildRoot, $RawWheelRoot, $RepairedWheelRoot, $PythonBuildRoot

$ArchiveValid = Test-Path -LiteralPath $OnnxArchive
if ($ArchiveValid) {
  $ArchiveValid = (Get-FileHash -LiteralPath $OnnxArchive -Algorithm SHA256).Hash -eq $OnnxArchiveSha256
}
if (-not $ArchiveValid) {
  if (Test-Path -LiteralPath $OnnxArchive) {
    Remove-Item -LiteralPath $OnnxArchive -Force
  }
  Invoke-WebRequest `
    -Uri "https://github.com/microsoft/onnxruntime/releases/download/v$OnnxVersion/$OnnxArchiveName" `
    -OutFile $OnnxArchive `
    -UseBasicParsing
  if ((Get-FileHash -LiteralPath $OnnxArchive -Algorithm SHA256).Hash -ne $OnnxArchiveSha256) {
    throw "The downloaded ONNX Runtime archive did not match its pinned SHA-256 digest."
  }
}
if (-not (Test-Path -LiteralPath (Join-Path $OnnxSourceRoot "include/onnxruntime_cxx_api.h"))) {
  Expand-Archive -LiteralPath $OnnxArchive -DestinationPath $BuildRoot -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $OnnxSourceRoot "include/onnxruntime_cxx_api.h")) -or
    -not (Test-Path -LiteralPath (Join-Path $OnnxLibRoot "onnxruntime.lib"))) {
  throw "The pinned ONNX Runtime archive has an unexpected layout."
}
# COLMAP's find module expects <hint>/onnxruntime/onnxruntime_cxx_api.h,
# whereas Microsoft's release archive stores headers directly in include/.
if (-not (Test-Path -LiteralPath (Join-Path $OnnxIncludeRoot "onnxruntime/onnxruntime_cxx_api.h"))) {
  $null = New-Item -ItemType Directory -Force -Path (Join-Path $OnnxIncludeRoot "onnxruntime")
  Copy-Item -Path (Join-Path $OnnxSourceRoot "include/*") -Destination (Join-Path $OnnxIncludeRoot "onnxruntime") -Recurse -Force
}

$env:Path = "$RuntimeScripts;$env:Path"
$env:VCPKG_DISABLE_METRICS = "1"
$env:VCPKG_FEATURE_FLAGS = "manifests,versions"
$env:CUDAFLAGS = "-allow-unsupported-compiler"
$env:CMAKE_BUILD_PARALLEL_LEVEL = [Math]::Max(1, [Math]::Min([Environment]::ProcessorCount, 16)).ToString()

if (-not $CudaArchitectures) {
  if ($Mode -eq "Native") {
    $CudaArchitectures = (& $Python -c "import torch; major, minor = torch.cuda.get_device_capability(0); print(f'{major}{minor}')").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $CudaArchitectures) {
      throw "Could not determine the native CUDA compute architecture."
    }
  } else {
    # CUDA 13 supports Turing through Blackwell. Keep the release set explicit
    # so future CMake architecture-list changes cannot silently alter packages.
    $CudaArchitectures = "75;86;89;120"
  }
}

$ExpectedStamp = "colmap=$ColmapVersion;mode=$Mode;cuda=$($env:CUDA_PATH);arch=$CudaArchitectures;onnx=$OnnxVersion-cuda13;wheel=onnx-providers-v1"
$InstalledStamp = if (Test-Path -LiteralPath $StampPath) {
  (Get-Content -Raw -LiteralPath $StampPath).Trim()
} else { "" }
if ($InstalledStamp -eq $ExpectedStamp) {
  & $Python -c "import pycolmap, sys; sys.exit(0 if pycolmap.__version__ == '$ColmapVersion' and pycolmap.has_cuda else 1)"
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Reusing CUDA-enabled PyCOLMAP: $ExpectedStamp"
    exit 0
  }
}

if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot ".git"))) {
  Invoke-Checked "COLMAP source checkout" {
    git clone --depth 1 --branch $ColmapVersion https://github.com/colmap/colmap.git $SourceRoot
  }
}
$SourceTag = (& git -C $SourceRoot describe --tags --exact-match 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceTag -ne $ColmapVersion) {
  throw "The cached COLMAP source is not the requested $ColmapVersion tag: $SourceRoot"
}

# COLMAP normally maps ONNX_ENABLED + FETCH_ONNX=OFF to vcpkg's `onnx`
# feature. That port tries to discover a separately installed cuDNN and cannot
# consume Microsoft's self-contained CUDA 13 release archive. Keep the upstream
# find_package path, but suppress only that inferred vcpkg feature so the pinned
# external runtime below is used directly.
$ColmapCmakeLists = Join-Path $SourceRoot "CMakeLists.txt"
$OriginalOnnxCondition = "if(ONNX_ENABLED AND NOT FETCH_ONNX)"
$ScanLanOnnxCondition = "if(ONNX_ENABLED AND NOT FETCH_ONNX AND NOT SCANLAN_EXTERNAL_ONNX)"
$ColmapCmakeContent = [System.IO.File]::ReadAllText($ColmapCmakeLists)
if ($ColmapCmakeContent.Contains($OriginalOnnxCondition)) {
  $ColmapCmakeContent = $ColmapCmakeContent.Replace($OriginalOnnxCondition, $ScanLanOnnxCondition)
  [System.IO.File]::WriteAllText(
    $ColmapCmakeLists,
    $ColmapCmakeContent,
    [System.Text.UTF8Encoding]::new($false)
  )
} elseif (-not $ColmapCmakeContent.Contains($ScanLanOnnxCondition)) {
  throw "COLMAP's ONNX vcpkg feature condition changed; update ScanLan's pinned build patch."
}

if (-not (Test-Path -LiteralPath (Join-Path $VcpkgRoot ".git"))) {
  Invoke-Checked "vcpkg checkout" {
    git clone --filter=blob:none https://github.com/microsoft/vcpkg.git $VcpkgRoot
  }
}
Invoke-Checked "vcpkg baseline fetch" {
  git -C $VcpkgRoot fetch origin $VcpkgBaseline --depth 1
}
Invoke-Checked "vcpkg baseline selection" {
  git -C $VcpkgRoot checkout --detach $VcpkgBaseline
}
if (-not (Test-Path -LiteralPath (Join-Path $VcpkgRoot "vcpkg.exe"))) {
  Invoke-Checked "vcpkg bootstrap" {
    & (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
  }
}

# Import the same Visual Studio developer environment used by COLMAP's Windows
# CI. Environment-variable changes made by the script remain process-wide.
& (Join-Path $SourceRoot "scripts/shell/enter_vs_dev_shell.ps1")

$Toolchain = (Join-Path $VcpkgRoot "scripts/buildsystems/vcpkg.cmake").Replace("\", "/")
$InstallRootCmake = $InstallRoot.Replace("\", "/")
$VcpkgInstalledCmake = $VcpkgInstalledRoot.Replace("\", "/")

$ConfigureArguments = @(
  "-S", $SourceRoot,
  "-B", $NativeBuildRoot,
  "-G", "Ninja",
  "-DCMAKE_MAKE_PROGRAM=$(Join-Path $RuntimeScripts 'ninja.exe')",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=$InstallRootCmake",
  "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
  "-DVCPKG_TARGET_TRIPLET=$Triplet",
  "-DVCPKG_INSTALLED_DIR=$VcpkgInstalledCmake",
  "-DCUDA_ENABLED=ON",
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
  "-DCUDAToolkit_ROOT=$($env:CUDA_PATH.Replace('\', '/'))",
  "-DGUI_ENABLED=OFF",
  "-DONNX_ENABLED=ON",
  "-DFETCH_ONNX=OFF",
  "-DSCANLAN_EXTERNAL_ONNX=ON",
  "-Donnxruntime_INCLUDE_DIR_HINTS=$($OnnxIncludeRoot.Replace('\', '/'))",
  "-Donnxruntime_LIBRARY_DIR_HINTS=$($OnnxLibRoot.Replace('\', '/'))",
  "-DCGAL_ENABLED=OFF",
  "-DLSD_ENABLED=OFF",
  "-DMVS_ENABLED=OFF",
  "-DDOWNLOAD_ENABLED=OFF",
  "-DTESTS_ENABLED=OFF",
  "-DBENCHMARK_ENABLED=OFF"
)
Invoke-Checked "CUDA COLMAP configuration" {
  & cmake @ConfigureArguments
}
Invoke-Checked "CUDA COLMAP build" {
  & cmake --build $NativeBuildRoot --target install --parallel $env:CMAKE_BUILD_PARALLEL_LEVEL
}

Invoke-Checked "PyCOLMAP build dependencies" {
  & $Python -m pip install --upgrade "scikit-build-core>=0.3.3" "pybind11==3.0.2" "delvewheel>=1.10"
}
$Pybind11CmakeDir = (& $Python -m pybind11 --cmakedir).Trim().Replace("\", "/")
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Pybind11CmakeDir)) {
  throw "The pybind11 CMake package directory could not be resolved."
}

$PythonBuildArguments = @(
  "-m", "pip", "wheel", "--verbose", $SourceRoot,
  "--no-deps",
  "--wheel-dir", $RawWheelRoot,
  "--config-settings", "build-dir=$($PythonBuildRoot.Replace('\', '/'))",
  "--config-settings", "cmake.build-type=Release",
  "--config-settings", "cmake.define.CMAKE_TOOLCHAIN_FILE=$Toolchain",
  "--config-settings", "cmake.define.VCPKG_TARGET_TRIPLET=$Triplet",
  "--config-settings", "cmake.define.VCPKG_INSTALLED_DIR=$VcpkgInstalledCmake",
  "--config-settings", "cmake.define.VCPKG_MANIFEST_MODE=OFF",
  "--config-settings", "cmake.define.colmap_DIR=$InstallRootCmake/share/colmap",
  "--config-settings", "cmake.define.CMAKE_PREFIX_PATH=$InstallRootCmake",
  "--config-settings", "cmake.define.pybind11_DIR=$Pybind11CmakeDir",
  "--config-settings", "cmake.define.CMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
  "--config-settings", "cmake.define.ONNX_ENABLED=ON",
  "--config-settings", "cmake.define.onnxruntime_INCLUDE_DIR_HINTS=$($OnnxIncludeRoot.Replace('\', '/'))",
  "--config-settings", "cmake.define.onnxruntime_LIBRARY_DIR_HINTS=$($OnnxLibRoot.Replace('\', '/'))",
  "--config-settings", "cmake.define.GENERATE_STUBS=OFF"
)
Invoke-Checked "CUDA PyCOLMAP wheel build" {
  & $Python @PythonBuildArguments
}

$RawWheel = Get-ChildItem -LiteralPath $RawWheelRoot -Filter "pycolmap-$ColmapVersion-*.whl" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not $RawWheel) {
  throw "The CUDA PyCOLMAP build did not produce a wheel."
}
$DependencyPaths = @(
  (Join-Path $VcpkgInstalledRoot "$Triplet/bin"),
  (Join-Path $InstallRoot "bin"),
  $OnnxLibRoot,
  (Join-Path $env:CUDA_PATH "bin")
) | Where-Object { Test-Path -LiteralPath $_ }
Invoke-Checked "CUDA PyCOLMAP dependency repair" {
  # ONNX loads execution-provider DLLs dynamically by their fixed basenames,
  # so dependency scanners cannot discover them and name mangling would make
  # them invisible at runtime. Force all three runtime DLLs into the wheel with
  # their upstream names; the pinned versions are identical and colocated.
  & $Python -m delvewheel repair $RawWheel.FullName `
    --add-path ($DependencyPaths -join ";") `
    --include "onnxruntime_providers_shared.dll;onnxruntime_providers_cuda.dll" `
    --no-mangle "onnxruntime.dll;onnxruntime_providers_shared.dll;onnxruntime_providers_cuda.dll" `
    --wheel-dir $RepairedWheelRoot
}

$RepairedWheel = Get-ChildItem -LiteralPath $RepairedWheelRoot -Filter "pycolmap-$ColmapVersion-*.whl" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not $RepairedWheel) {
  throw "Dependency repair did not produce a CUDA PyCOLMAP wheel."
}
Invoke-Checked "CUDA PyCOLMAP installation" {
  & $Python -m pip install --upgrade --force-reinstall --no-deps $RepairedWheel.FullName
}
# delvewheel's build-time pefile requirement currently selects the one release
# excluded by PyInstaller. Restore the runtime packager's compatible version
# after DLL repair has completed.
Invoke-Checked "PyInstaller dependency restoration" {
  & $Python -m pip install --force-reinstall "pefile==2023.2.7"
}
Invoke-Checked "CUDA wheel-tool cleanup" {
  & $Python -m pip uninstall --yes delvewheel
}

$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StampPath)
Set-Content -LiteralPath $StampPath -Value $ExpectedStamp -Encoding ASCII
Invoke-Checked "CUDA PyCOLMAP validation" {
  & $Python -c "import pycolmap; e=pycolmap.FeatureExtractionOptions(); m=pycolmap.FeatureMatchingOptions(); assert pycolmap.__version__ == '$ColmapVersion'; assert pycolmap.has_cuda; assert hasattr(e, 'aliked'); assert hasattr(m.sift, 'lightglue'); print('CUDA + ONNX PyCOLMAP', pycolmap.__version__, 'architectures', '$CudaArchitectures')"
}

Write-Host "CUDA PyCOLMAP ready: $($RepairedWheel.FullName)"
