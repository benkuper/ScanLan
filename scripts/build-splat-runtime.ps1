$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PackageRoot = Join-Path $ProjectRoot "splat-worker"
$RuntimeRoot = Join-Path $PackageRoot ".venv"
$Python = Join-Path $RuntimeRoot "Scripts/python.exe"
$RuntimeScripts = Join-Path $RuntimeRoot "Scripts"
$GsplatExtension = Join-Path $RuntimeRoot "Lib/site-packages/gsplat/csrc.pyd"

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

& $Python -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "Could not update the splat runtime installer." }
& $Python -c "import torch; assert torch.__version__.startswith('2.12.0+cu130'); assert torch.cuda.is_available()" 2>$null
if ($LASTEXITCODE -ne 0) {
  & $Python -m pip install --upgrade --force-reinstall "torch==2.12.0" --index-url "https://download.pytorch.org/whl/cu130"
  if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch could not be installed." }
}
& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "The installed PyTorch runtime cannot access CUDA." }
$CudaArchitecture = (& $Python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))").Trim()
$env:TORCH_CUDA_ARCH_LIST = $CudaArchitecture
& $Python -m pip install --upgrade "numpy==1.26.4" "Pillow==11.1.0" "ninja>=1.10" "jaxtyping" "rich>=12" "backports.tarfile" "pyinstaller==6.16.0"
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
if (-not (Test-Path -LiteralPath $GsplatExtension)) {
  & $Python -m pip install --upgrade --force-reinstall --no-deps --only-binary=:all: "gsplat==1.5.3"
  if ($LASTEXITCODE -ne 0) { throw "gsplat could not be installed." }
  & $Python (Join-Path $PackageRoot "build_gsplat_extension.py")
  if ($LASTEXITCODE -ne 0) { throw "The 3D gsplat CUDA extension could not be compiled for architecture $CudaArchitecture." }
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
    --collect-all gsplat --copy-metadata torch --copy-metadata gsplat entry.py
  if ($LASTEXITCODE -ne 0) { throw "ScanLan splat runtime packaging failed." }
} finally {
  Pop-Location
}
$PackagedWorker = Join-Path $PackageRoot "dist/scanlan-splat/scanlan-splat.exe"
& $PackagedWorker diagnostics --require-cuda
if ($LASTEXITCODE -ne 0) { throw "The packaged ScanLan splat runtime failed validation." }

Write-Host "Splat runtime ready: $PackagedWorker"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Warning "FFmpeg is missing; RGB-D splats work, but video import does not." }
if (-not (Get-Command colmap -ErrorAction SilentlyContinue)) { Write-Warning "COLMAP is missing; RGB-D splats work, but photo/video registration does not." }
