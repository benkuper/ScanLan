$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PackageRoot = Join-Path $ProjectRoot "splat-worker"
$RuntimeRoot = Join-Path $PackageRoot ".venv"
$Python = Join-Path $RuntimeRoot "Scripts/python.exe"
$RuntimeScripts = Join-Path $RuntimeRoot "Scripts"
$SitePackages = Join-Path $RuntimeRoot "Lib/site-packages"
$GsplatExtension = Join-Path $RuntimeRoot "Lib/site-packages/gsplat/csrc.pyd"
$GsplatFeatureStamp = Join-Path $RuntimeRoot "Lib/site-packages/gsplat/scanlan-build.txt"
$GsplatFeatureSchema = "2dgs-rgbd-v3"

if (-not (Test-Path -LiteralPath $Python)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -m venv $RuntimeRoot
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; assert sys.version_info[:2] == (3, 11), 'ScanLan splats require Python 3.11'"
    & python -m venv $RuntimeRoot
  } else {
    throw "Python 3.11 was not found. Install it before preparing Gaussian-splat support."
  }
}

if (Test-Path -LiteralPath $SitePackages) {
  $StalePipDistributions = @(Get-ChildItem -LiteralPath $SitePackages -Force |
    Where-Object { $_.PSIsContainer -and ($_.Name -eq "~ip" -or $_.Name -like "~ip-*.dist-info") })
  foreach ($Distribution in $StalePipDistributions) {
    Write-Host "Removing stale pip upgrade artifact: $($Distribution.Name)"
    Remove-Item -LiteralPath $Distribution.FullName -Recurse -Force
  }
}

& $Python -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "Could not update the splat runtime installer." }
$PreviousErrorActionPreference = $ErrorActionPreference
try {
  # This is an expected-failure probe: a fresh environment does not have torch yet.
  # Windows PowerShell promotes native stderr to NativeCommandError when the global
  # preference is Stop, so suppress it until the probe exit code has been captured.
  $ErrorActionPreference = "SilentlyContinue"
  & $Python -c "import importlib.util, sys; found = importlib.util.find_spec('torch') is not None; sys.exit(1) if not found else None; import torch; sys.exit(0 if torch.__version__.startswith('2.12.1+cu130') and torch.cuda.is_available() else 1)" 2>$null
  $TorchReady = $LASTEXITCODE -eq 0
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
if (-not $TorchReady) {
  & $Python -m pip install --upgrade --force-reinstall "torch==2.12.1" --index-url "https://download.pytorch.org/whl/cu130"
  if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch could not be installed." }
}
& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "The installed PyTorch runtime cannot access CUDA." }
$CudaArchitecture = (& $Python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))").Trim()
$TorchBuild = (& $Python -c "import torch; print(torch.__version__)").Trim()
$ExpectedGsplatFeatures = "$GsplatFeatureSchema;torch=$TorchBuild;arch=$CudaArchitecture"
$env:TORCH_CUDA_ARCH_LIST = $CudaArchitecture
& $Python -m pip install --upgrade "numpy==1.26.4" "Pillow==11.1.0" "pycolmap==4.1.1" "av==18.0.0" "ninja>=1.10" "jaxtyping" "rich>=12" "backports.tarfile" "pyinstaller==6.16.0"
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat support dependencies could not be installed." }
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$VisualStudioRoot = if (Test-Path -LiteralPath $VsWhere) {
  & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
} else { $null }
$VcVars = if ($VisualStudioRoot) { Join-Path $VisualStudioRoot "VC/Auxiliary/Build/vcvars64.bat" } else { $null }
if (-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)) {
  throw "Visual Studio C++ build tools were not found for the gsplat CUDA extension."
}
$VcArguments = "`"$VcVars`" >nul && set"
$VcEnvironment = & cmd.exe /d /s /c $VcArguments
if ($LASTEXITCODE -ne 0) { throw "Visual Studio C++ build environment could not be initialized." }
foreach ($Entry in $VcEnvironment) {
  if ($Entry -match '^([^=]+)=(.*)$') {
    Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
  }
}
$env:PATH = $RuntimeScripts + [IO.Path]::PathSeparator + $env:PATH
$env:CL = (($env:CL + " /Zc:preprocessor").Trim())
$env:MAX_JOBS = [Math]::Min([Environment]::ProcessorCount, 4).ToString()
$InstalledGsplatFeatures = if (Test-Path -LiteralPath $GsplatFeatureStamp) {
  (Get-Content -Raw -LiteralPath $GsplatFeatureStamp).Trim()
} else { "" }
if ((Test-Path -LiteralPath $GsplatExtension) -and $InstalledGsplatFeatures -ne $ExpectedGsplatFeatures) {
  Write-Host "Rebuilding gsplat with ScanLan 2DGS kernels."
  Remove-Item -LiteralPath $GsplatExtension -Force
}
if (-not (Test-Path -LiteralPath $GsplatExtension)) {
  & $Python -m pip install --upgrade --force-reinstall --no-deps --only-binary=:all: "gsplat==1.5.3"
  if ($LASTEXITCODE -ne 0) { throw "gsplat could not be installed." }
  & $Python (Join-Path $PackageRoot "build_gsplat_extension.py")
  if ($LASTEXITCODE -ne 0) { throw "The gsplat 2DGS CUDA extension could not be compiled for architecture $CudaArchitecture." }
  Set-Content -LiteralPath $GsplatFeatureStamp -Value $ExpectedGsplatFeatures -Encoding ASCII
} else {
  Write-Host "Reusing gsplat CUDA extension: $GsplatExtension"
}
& $Python -m pip install --upgrade --force-reinstall --no-deps $PackageRoot
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat worker could not be installed." }
& $Python -m scanlan_splat.cli diagnostics --require-cuda
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat runtime validation failed." }

Push-Location $PackageRoot
try {
  & $Python -m PyInstaller --noconfirm --clean --onedir --name scanlan-splat `
    --hidden-import gsplat.csrc --hidden-import backports.tarfile `
    --add-binary "${GsplatExtension};gsplat" `
    --collect-all gsplat --collect-all pycolmap --collect-all av `
    --copy-metadata torch --copy-metadata gsplat --copy-metadata pycolmap --copy-metadata av entry.py
  if ($LASTEXITCODE -ne 0) { throw "ScanLan splat runtime packaging failed." }
} finally {
  Pop-Location
}
$PackagedWorker = Join-Path $PackageRoot "dist/scanlan-splat/scanlan-splat.exe"
& $PackagedWorker diagnostics --require-cuda
if ($LASTEXITCODE -ne 0) { throw "The packaged ScanLan splat runtime failed validation." }

Write-Host "Splat runtime ready: $PackagedWorker"
