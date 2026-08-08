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
$NativeBuildRoot = Join-Path $BuildRoot "native-$ColmapVersion-$($Mode.ToLowerInvariant())"
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
Assert-ChildPath $BuildRoot $NativeBuildRoot
$null = New-Item -ItemType Directory -Force -Path $BuildRoot, $NativeBuildRoot, $RawWheelRoot, $RepairedWheelRoot, $PythonBuildRoot

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

$ExpectedStamp = "colmap=$ColmapVersion;mode=$Mode;cuda=$($env:CUDA_PATH);arch=$CudaArchitectures"
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
  "-DONNX_ENABLED=OFF",
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
  (Join-Path $env:CUDA_PATH "bin")
) | Where-Object { Test-Path -LiteralPath $_ }
Invoke-Checked "CUDA PyCOLMAP dependency repair" {
  & $Python -m delvewheel repair $RawWheel.FullName --add-path ($DependencyPaths -join ";") --wheel-dir $RepairedWheelRoot
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
  & $Python -c "import pycolmap; assert pycolmap.__version__ == '$ColmapVersion'; assert pycolmap.has_cuda; print('CUDA PyCOLMAP', pycolmap.__version__, 'architectures', '$CudaArchitectures')"
}

Write-Host "CUDA PyCOLMAP ready: $($RepairedWheel.FullName)"
