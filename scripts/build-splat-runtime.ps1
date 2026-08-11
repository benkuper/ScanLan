param(
  [ValidateSet("Native", "Redistributable")]
  [string]$PycolmapMode = "Native"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PackageRoot = Join-Path $ProjectRoot "splat-worker"
$GeometryPackageRoot = Join-Path $ProjectRoot "geometry-worker"
$RuntimeRoot = Join-Path $PackageRoot ".venv"
$Python = Join-Path $RuntimeRoot "Scripts/python.exe"
$RuntimeScripts = Join-Path $RuntimeRoot "Scripts"
$SitePackages = Join-Path $RuntimeRoot "Lib/site-packages"
$GsplatExtension = Join-Path $RuntimeRoot "Lib/site-packages/gsplat/csrc.pyd"
$PycolmapLibraries = Join-Path $SitePackages "pycolmap.libs"
$PycolmapOnnxShared = Join-Path $PycolmapLibraries "onnxruntime_providers_shared.dll"
$PycolmapOnnxCuda = Join-Path $PycolmapLibraries "onnxruntime_providers_cuda.dll"
$GsplatFeatureStamp = Join-Path $RuntimeRoot "Lib/site-packages/gsplat/scanlan-build.txt"
$GsplatFeatureSchema = "2dgs-rgbd-v3"
$LingbotRevision = "1f480aeb8a47a24656090d46d053115b7fe60435"
$LingbotModelRevision = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
$LingbotDepthRevision = "f3a237e434ae987bc38281476d6cfb5df3e4d739"
$LingbotDepthModelRevision = "79204ed6b837f4fdd192cf563e59481fecfa0295"
$LingbotDepthModelName = "lingbot-depth-v0.5.pt"
$LingbotDepthModelSha256 = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"
$MapAnythingRevision = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
$MapAnythingModelRevision = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
$MapAnythingModelSha256 = "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
$Da3Revision = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
$Da3ModelRevision = "b2359bdf726fb44ef62acca04d629dcf158053e7"
$Da3ModelSha256 = "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
$Da3ConfigSha256 = "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
$FlashInferRevision = "713358284345314df4f40ddc352f4e981f5bb03e"
$FlashInferFeatureStamp = Join-Path $SitePackages "flashinfer/scanlan-build.txt"
$LingbotModels = Join-Path $PackageRoot "models"
$MapAnythingModels = Join-Path $LingbotModels "map-anything-apache"
$Da3Models = Join-Path $LingbotModels "da3nested-giant-large-1.1-noncommercial"
$LingbotModelAssets = @(
  @{
    Name = "lingbot-map-long.pt"
    Sha256 = "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
  },
  @{
    Name = "skyseg_batch.onnx"
    Sha256 = "b09c0f6cf79e1caa2591b946b659487bd7c8208caddd3f80680cbb169617e378"
  }
)
$FlashInferSource = Join-Path $PackageRoot "build/flashinfer-windows"
$FlashInferCacheArchive = Join-Path $LingbotModels "flashinfer-cache.zip"
$ColmapModelRelease = "3.13.0"
$ColmapModelAssets = @(
  @{
    Name = "aliked-n16rot.onnx"
    Sha256 = "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547"
  },
  @{
    Name = "aliked-lightglue.onnx"
    Sha256 = "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d"
  },
  @{
    Name = "sift-lightglue.onnx"
    Sha256 = "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e"
  }
)

function Get-VerifiedDownload([string]$Uri, [string]$Destination, [string]$Sha256) {
  $Valid = Test-Path -LiteralPath $Destination
  if ($Valid) {
    $Valid = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Sha256
  }
  if (-not $Valid) {
    if (Test-Path -LiteralPath $Destination) {
      Remove-Item -LiteralPath $Destination -Force
    }
    Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $Sha256) {
      throw "Downloaded model $([System.IO.Path]::GetFileName($Destination)) did not match its pinned SHA-256 digest."
    }
  }
}

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
& $Python -m pip install --upgrade --no-deps "torchvision==0.27.1" --index-url "https://download.pytorch.org/whl/cu130"
if ($LASTEXITCODE -ne 0) { throw "The LingBot-compatible torchvision runtime could not be installed." }
& $Python -m pip install --upgrade `
  "numpy==1.26.4" "Pillow==11.1.0" "av==18.0.0" "ninja>=1.10" `
  "jaxtyping" "rich>=12" "backports.tarfile" "pyinstaller==6.16.0" `
  "huggingface-hub>=0.34,<2" "einops>=0.8,<1" "safetensors>=0.5,<1" `
  "opencv-python==4.11.0.86" "scipy>=1.15,<2" "tqdm>=4.67,<5" `
  "click>=8.1,<9" "matplotlib>=3.10,<4" "trimesh>=4.8,<5" `
  "onnxruntime-gpu==1.28.0" "hydra-core==1.3.5" "natsort==8.4.0" `
  "orjson==3.11.9" "pillow-heif==1.5.0" "python-box==7.4.1" `
  "termcolor==3.3.0" "timm==1.0.28" "addict>=2.4,<3" "e3nn>=0.5,<1" `
  "plyfile>=1.1,<2" "evo>=1.33,<2" "imageio>=2.37,<3" "moviepy==1.0.3"
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat support dependencies could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps `
  "git+https://github.com/Robbyant/lingbot-map.git@$LingbotRevision"
if ($LASTEXITCODE -ne 0) { throw "LingBot-Map could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps `
  "git+https://github.com/Robbyant/lingbot-depth.git@$LingbotDepthRevision"
if ($LASTEXITCODE -ne 0) { throw "LingBot-Depth could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps "uniception==0.1.7"
if ($LASTEXITCODE -ne 0) { throw "UniCeption 0.1.7 could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps `
  "git+https://github.com/facebookresearch/map-anything.git@$MapAnythingRevision"
if ($LASTEXITCODE -ne 0) { throw "MapAnything could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps `
  "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@$Da3Revision"
if ($LASTEXITCODE -ne 0) { throw "Depth Anything 3 could not be installed." }

# FlashInfer's upstream wheels do not support native Windows. Build the pinned
# Windows fork's JIT package for the exact Torch/CUDA/GPU combination. It keeps
# LingBot's long-video cache bounded and is required by ScanLan's quality path.
$ExpectedFlashInferBuild = "revision=$FlashInferRevision;torch=$TorchBuild;arch=$CudaArchitecture"
$InstalledFlashInferBuild = if (Test-Path -LiteralPath $FlashInferFeatureStamp) {
  (Get-Content -Raw -LiteralPath $FlashInferFeatureStamp).Trim()
} else { "" }
$FlashInferReady = $false
$PreviousErrorActionPreference = $ErrorActionPreference
try {
  $ErrorActionPreference = "SilentlyContinue"
  & $Python -c "import flashinfer; print(flashinfer.__version__)" 2>$null
  $FlashInferReady = $LASTEXITCODE -eq 0 -and $InstalledFlashInferBuild -eq $ExpectedFlashInferBuild
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
if (-not $FlashInferReady) {
  $FlashInferBuildOk = $true
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    # Every command in this block is optional. Preserve its exit status without
    # allowing native stderr to abort the required SDPA-capable runtime build.
    $ErrorActionPreference = "Continue"
    if (Test-Path -LiteralPath $FlashInferSource) {
      $InstalledFlashInferRevision = (& git -C $FlashInferSource rev-parse HEAD).Trim()
      if ($LASTEXITCODE -ne 0 -or $InstalledFlashInferRevision -ne $FlashInferRevision) {
        Remove-Item -LiteralPath $FlashInferSource -Recurse -Force
      }
    }
    if (-not (Test-Path -LiteralPath $FlashInferSource)) {
      & git clone --no-checkout https://github.com/SystemPanic/flashinfer-windows.git $FlashInferSource
      $FlashInferBuildOk = $LASTEXITCODE -eq 0
      if ($FlashInferBuildOk) {
        & git -C $FlashInferSource checkout --detach $FlashInferRevision
        $FlashInferBuildOk = $LASTEXITCODE -eq 0
      }
      if ($FlashInferBuildOk) {
        & git -C $FlashInferSource submodule update --init --recursive
        $FlashInferBuildOk = $LASTEXITCODE -eq 0
      }
    }
    if ($FlashInferBuildOk) {
      & $Python -m pip install --upgrade `
        "apache-tvm-ffi>=0.1.9,<0.2" click cuda-tile nvidia-cudnn-frontend `
        nvidia-ml-py "packaging>=24.2" requests tabulate
      $FlashInferBuildOk = $LASTEXITCODE -eq 0
    }
    if ($FlashInferBuildOk) {
      $env:FLASHINFER_CUDA_ARCH_LIST = $CudaArchitecture
      & $Python -m pip install --upgrade --force-reinstall --no-build-isolation --no-deps $FlashInferSource
      $FlashInferBuildOk = $LASTEXITCODE -eq 0
    }
    if ($FlashInferBuildOk) {
      & $Python -c "import flashinfer; print(flashinfer.__version__)"
      $FlashInferReady = $LASTEXITCODE -eq 0
      if ($FlashInferReady) {
        Set-Content -LiteralPath $FlashInferFeatureStamp -Value $ExpectedFlashInferBuild -Encoding ASCII
      }
    }
  } catch {
    $FlashInferBuildOk = $false
    Write-Warning "Windows FlashInfer build failed: $($_.Exception.Message)"
  } finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  if (-not $FlashInferReady) {
    throw "The pinned Windows FlashInfer runtime could not be built."
  }
}

# Resolve immutable model assets once during runtime preparation. A verified
# local copy is authoritative, avoiding a 4.6 GB cache/network lookup on every
# release. Missing or mismatched files are downloaded from the pinned revision.
$null = New-Item -ItemType Directory -Path $LingbotModels -Force
foreach ($Asset in $LingbotModelAssets) {
  $Destination = Join-Path $LingbotModels $Asset.Name
  $Valid = Test-Path -LiteralPath $Destination
  if ($Valid) {
    $Valid = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Asset.Sha256
  }
  if (-not $Valid) {
    if (Test-Path -LiteralPath $Destination) {
      Remove-Item -LiteralPath $Destination -Force
    }
    $Downloaded = (& $Python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='robbyant/lingbot-map', filename='$($Asset.Name)', revision='$LingbotModelRevision'))" | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Downloaded)) {
      throw "LingBot-Map asset $($Asset.Name) could not be downloaded."
    }
    try {
      $null = New-Item -ItemType HardLink -Path $Destination -Target $Downloaded
    } catch {
      Copy-Item -LiteralPath $Downloaded -Destination $Destination
    }
  }
  if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $Asset.Sha256) {
    throw "LingBot-Map asset $($Asset.Name) did not match its pinned SHA-256 digest."
  }
}
$LingbotDepthDestination = Join-Path $LingbotModels $LingbotDepthModelName
$LingbotDepthValid = Test-Path -LiteralPath $LingbotDepthDestination
if ($LingbotDepthValid) {
  $LingbotDepthValid = (Get-FileHash -LiteralPath $LingbotDepthDestination -Algorithm SHA256).Hash -eq $LingbotDepthModelSha256
}
if (-not $LingbotDepthValid) {
  if (Test-Path -LiteralPath $LingbotDepthDestination) {
    Remove-Item -LiteralPath $LingbotDepthDestination -Force
  }
  $LingbotDepthDownloaded = (& $Python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='robbyant/lingbot-depth-pretrain-vitl-14-v0.5', filename='model.pt', revision='$LingbotDepthModelRevision'))" | Select-Object -Last 1)
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $LingbotDepthDownloaded)) {
    throw "LingBot-Depth v0.5 model could not be downloaded."
  }
  try {
    $null = New-Item -ItemType HardLink -Path $LingbotDepthDestination -Target $LingbotDepthDownloaded
  } catch {
    Copy-Item -LiteralPath $LingbotDepthDownloaded -Destination $LingbotDepthDestination
  }
}
if ((Get-FileHash -LiteralPath $LingbotDepthDestination -Algorithm SHA256).Hash -ne $LingbotDepthModelSha256) {
  throw "LingBot-Depth v0.5 did not match its pinned SHA-256 digest."
}
$null = New-Item -ItemType Directory -Path $MapAnythingModels -Force
foreach ($MapAnythingAsset in @("config.json", "model.safetensors")) {
  $MapAnythingDestination = Join-Path $MapAnythingModels $MapAnythingAsset
  $MapAnythingValid = Test-Path -LiteralPath $MapAnythingDestination
  if ($MapAnythingValid -and $MapAnythingAsset -eq "model.safetensors") {
    $MapAnythingValid = (Get-FileHash -LiteralPath $MapAnythingDestination -Algorithm SHA256).Hash -eq $MapAnythingModelSha256
  }
  if (-not $MapAnythingValid) {
    if (Test-Path -LiteralPath $MapAnythingDestination) {
      Remove-Item -LiteralPath $MapAnythingDestination -Force
    }
    $MapAnythingDownloaded = (& $Python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='facebook/map-anything-apache', filename='$MapAnythingAsset', revision='$MapAnythingModelRevision'))" | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $MapAnythingDownloaded)) {
      throw "MapAnything asset $MapAnythingAsset could not be downloaded."
    }
    Copy-Item -LiteralPath $MapAnythingDownloaded -Destination $MapAnythingDestination -Force
  }
}
if ((Get-FileHash -LiteralPath (Join-Path $MapAnythingModels "model.safetensors") -Algorithm SHA256).Hash -ne $MapAnythingModelSha256) {
  throw "MapAnything Apache did not match its pinned SHA-256 digest."
}
$null = New-Item -ItemType Directory -Path $Da3Models -Force
foreach ($Da3Asset in @("config.json", "model.safetensors")) {
  $Da3Destination = Join-Path $Da3Models $Da3Asset
  $Da3Valid = Test-Path -LiteralPath $Da3Destination
  if ($Da3Valid) {
    $Da3ExpectedSha256 = if ($Da3Asset -eq "model.safetensors") {
      $Da3ModelSha256
    } else {
      $Da3ConfigSha256
    }
    $Da3Valid = (Get-FileHash -LiteralPath $Da3Destination -Algorithm SHA256).Hash -eq $Da3ExpectedSha256
  }
  if (-not $Da3Valid) {
    if (Test-Path -LiteralPath $Da3Destination) {
      Remove-Item -LiteralPath $Da3Destination -Force
    }
    $Da3Downloaded = (& $Python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='depth-anything/DA3NESTED-GIANT-LARGE-1.1', filename='$Da3Asset', revision='$Da3ModelRevision'))" | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Da3Downloaded)) {
      throw "DA3 Nested Giant-Large 1.1 asset $Da3Asset could not be downloaded."
    }
    Copy-Item -LiteralPath $Da3Downloaded -Destination $Da3Destination -Force
  }
}
if ((Get-FileHash -LiteralPath (Join-Path $Da3Models "model.safetensors") -Algorithm SHA256).Hash -ne $Da3ModelSha256) {
  throw "DA3 Nested Giant-Large 1.1 did not match its pinned SHA-256 digest."
}
if ((Get-FileHash -LiteralPath (Join-Path $Da3Models "config.json") -Algorithm SHA256).Hash -ne $Da3ConfigSha256) {
  throw "DA3 Nested Giant-Large 1.1 configuration did not match its pinned SHA-256 digest."
}
foreach ($Asset in $ColmapModelAssets) {
  $Destination = Join-Path $LingbotModels $Asset.Name
  Get-VerifiedDownload `
    -Uri "https://github.com/colmap/colmap/releases/download/$ColmapModelRelease/$($Asset.Name)" `
    -Destination $Destination `
    -Sha256 $Asset.Sha256
}
$env:SCANLAN_LINGBOT_MODEL = Join-Path $LingbotModels "lingbot-map-long.pt"
$env:SCANLAN_LINGBOT_SKY_MODEL = Join-Path $LingbotModels "skyseg_batch.onnx"
$env:SCANLAN_LINGBOT_DEPTH_MODEL = $LingbotDepthDestination
$env:SCANLAN_MAPANYTHING_MODEL = $MapAnythingModels
$env:SCANLAN_DA3_MODEL = $Da3Models
$env:SCANLAN_COLMAP_ALIKED_MODEL = Join-Path $LingbotModels "aliked-n16rot.onnx"
$env:SCANLAN_COLMAP_LIGHTGLUE_MODEL = Join-Path $LingbotModels "aliked-lightglue.onnx"
$env:SCANLAN_FLASHINFER_CACHE_ARCHIVE = $FlashInferCacheArchive
& (Join-Path $PSScriptRoot "build-pycolmap-cuda.ps1") -Mode $PycolmapMode
if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyCOLMAP could not be built or validated." }
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
$env:DISTUTILS_USE_SDK = "1"
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
& $Python -m pip install --upgrade --force-reinstall --no-deps (Join-Path $ProjectRoot "validation")
if ($LASTEXITCODE -ne 0) { throw "Shared ScanLan validation engine could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps (Join-Path $ProjectRoot "material")
if ($LASTEXITCODE -ne 0) { throw "Shared ScanLan material foundation could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps $PackageRoot
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat worker could not be installed." }
& $Python -m pip install --upgrade --force-reinstall --no-deps $GeometryPackageRoot
if ($LASTEXITCODE -ne 0) { throw "ScanLan geometry worker could not be installed." }
if ($FlashInferReady) {
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    # The Windows fork uses C:\_fij deliberately: generated CUDA object names
    # exceed MAX_PATH when the cache is nested under a normal project path.
    Remove-Item Env:FLASHINFER_CACHE_DIR -ErrorAction SilentlyContinue
    & $Python -c "import json; from scanlan_splat.lingbot import warm_lingbot_flashinfer; print(json.dumps(warm_lingbot_flashinfer()))"
    $FlashInferReady = $LASTEXITCODE -eq 0
    if ($FlashInferReady) {
      $FlashInferWorkspace = (& $Python -c "from flashinfer.jit import env; print(env.FLASHINFER_WORKSPACE_DIR)" | Select-Object -Last 1).Trim()
      $FlashInferCompiledModules = Join-Path $FlashInferWorkspace "cached_ops"
      $FlashInferAotStaging = Join-Path $LingbotModels "flashinfer-aot"
      $CompiledFlashInferDlls = @(
        Get-ChildItem -LiteralPath $FlashInferCompiledModules -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue
      )
      if (-not $CompiledFlashInferDlls.Count) {
        $FlashInferReady = $false
      } else {
        $null = New-Item -ItemType Directory -Path $FlashInferAotStaging -Force
        foreach ($CompiledDll in $CompiledFlashInferDlls) {
          $ModuleDestination = Join-Path $FlashInferAotStaging $CompiledDll.Directory.Name
          $null = New-Item -ItemType Directory -Path $ModuleDestination -Force
          Copy-Item -LiteralPath $CompiledDll.FullName -Destination $ModuleDestination -Force
        }
        if (Test-Path -LiteralPath $FlashInferCacheArchive) {
          Remove-Item -LiteralPath $FlashInferCacheArchive -Force
        }
        Compress-Archive -LiteralPath $FlashInferAotStaging -DestinationPath $FlashInferCacheArchive -CompressionLevel Optimal
      }
    }
  } catch {
    $FlashInferReady = $false
    Write-Warning "Windows FlashInfer kernel warmup failed: $($_.Exception.Message)"
  } finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  if (-not $FlashInferReady) {
    throw "Windows FlashInfer could not execute LingBot's paged-attention validation shape."
  }
}
& $Python -m scanlan_splat.cli diagnostics `
  --require-cuda `
  --require-learned-features `
  --require-adaptive-frames
if ($LASTEXITCODE -ne 0) { throw "ScanLan splat runtime validation failed." }
& $Python -m scanlan_geometry.cli diagnostics `
  --require-lingbot `
  --require-lingbot-depth `
  --require-mapanything `
  --require-da3 `
  --require-flashinfer
if ($LASTEXITCODE -ne 0) { throw "ScanLan geometry runtime validation failed." }

Push-Location $PackageRoot
try {
  if (-not (Test-Path -LiteralPath $PycolmapOnnxShared) -or
      -not (Test-Path -LiteralPath $PycolmapOnnxCuda)) {
    throw "PyCOLMAP's ONNX CUDA provider DLLs are missing from the validated runtime."
  }
  $PyInstallerArguments = @(
    "--noconfirm", "--clean", "--onedir", "--name", "scanlan-splat",
    "--hidden-import", "gsplat.csrc", "--hidden-import", "backports.tarfile",
    "--add-binary", "${GsplatExtension};gsplat",
    # PyInstaller sees the ONNX provider DLLs as delay-loaded and otherwise
    # drops them. COLMAP resolves these fixed names beside onnxruntime.dll.
    "--add-binary", "${PycolmapOnnxShared};pycolmap.libs",
    "--add-binary", "${PycolmapOnnxCuda};pycolmap.libs",
    "--collect-all", "gsplat", "--collect-all", "pycolmap", "--collect-all", "av",
    "--collect-all", "onnxruntime", "--collect-all", "cv2",
    "--collect-all", "scanlan_material",
    "--collect-all", "scanlan_validation",
    "--collect-all", "huggingface_hub", "--collect-all", "tvm_ffi",
    "--collect-all", "ninja",
    "--copy-metadata", "torch",
    "--copy-metadata", "gsplat", "--copy-metadata", "pycolmap",
    "--copy-metadata", "av",
    "--exclude-module", "lingbot_map", "--exclude-module", "mdm", "--exclude-module", "flashinfer"
  )
  $PyInstallerArguments += "entry.py"
  & $Python -m PyInstaller @PyInstallerArguments
  if ($LASTEXITCODE -ne 0) { throw "ScanLan splat runtime packaging failed." }
} finally {
  Pop-Location
}
$PackagedWorker = Join-Path $PackageRoot "dist/scanlan-splat/scanlan-splat.exe"
$PackagedModels = Join-Path $PackageRoot "dist/scanlan-splat/models"
$null = New-Item -ItemType Directory -Path $PackagedModels -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") -Destination (Split-Path -Parent $PackagedModels) -Force
foreach ($Asset in $ColmapModelAssets) {
  Copy-Item -LiteralPath (Join-Path $LingbotModels $Asset.Name) -Destination $PackagedModels -Force
}

Push-Location $GeometryPackageRoot
try {
  $GeometryArguments = @(
    "--noconfirm", "--clean", "--onedir", "--name", "scanlan-geometry",
    "--collect-all", "lingbot_map", "--collect-all", "mdm", "--collect-all", "torchvision",
    "--collect-all", "mapanything", "--collect-all", "uniception",
    "--collect-all", "depth_anything_3",
    "--exclude-module", "depth_anything_3.bench",
    "--exclude-module", "depth_anything_3.app",
    "--exclude-module", "depth_anything_3.services",
    "--exclude-module", "depth_anything_3.cli",
    "--collect-all", "addict", "--collect-all", "e3nn", "--collect-all", "plyfile",
    "--collect-all", "evo", "--collect-all", "imageio", "--collect-all", "moviepy",
    "--collect-all", "safetensors",
    "--collect-all", "onnxruntime", "--collect-all", "cv2",
    "--collect-all", "scanlan_material",
    "--collect-all", "scanlan_validation",
    "--collect-all", "huggingface_hub", "--collect-all", "tvm_ffi", "--collect-all", "ninja",
    "--copy-metadata", "torch", "--copy-metadata", "torchvision",
    "--copy-metadata", "lingbot-map", "--copy-metadata", "mdm",
    "--copy-metadata", "mapanything", "--copy-metadata", "uniception",
    "--copy-metadata", "depth-anything-3", "--copy-metadata", "addict", "--copy-metadata", "e3nn", "--copy-metadata", "plyfile",
    "--copy-metadata", "safetensors"
  )
  if ($FlashInferReady) {
    $GeometryArguments += @("--collect-all", "flashinfer")
  }
  $GeometryArguments += "entry.py"
  & $Python -m PyInstaller @GeometryArguments
  if ($LASTEXITCODE -ne 0) { throw "ScanLan geometry runtime packaging failed." }
} finally {
  Pop-Location
}
$PackagedGeometryWorker = Join-Path $GeometryPackageRoot "dist/scanlan-geometry/scanlan-geometry.exe"
$PackagedGeometryModels = Join-Path $GeometryPackageRoot "dist/scanlan-geometry/models"
$null = New-Item -ItemType Directory -Path $PackagedGeometryModels -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") -Destination (Split-Path -Parent $PackagedGeometryModels) -Force
Copy-Item -LiteralPath (Join-Path $LingbotModels "lingbot-map-long.pt") -Destination $PackagedGeometryModels -Force
Copy-Item -LiteralPath (Join-Path $LingbotModels "skyseg_batch.onnx") -Destination $PackagedGeometryModels -Force
Copy-Item -LiteralPath $LingbotDepthDestination -Destination $PackagedGeometryModels -Force
$PackagedMapAnythingModels = Join-Path $PackagedGeometryModels "map-anything-apache"
$null = New-Item -ItemType Directory -Path $PackagedMapAnythingModels -Force
Copy-Item -LiteralPath (Join-Path $MapAnythingModels "config.json") -Destination $PackagedMapAnythingModels -Force
Copy-Item -LiteralPath (Join-Path $MapAnythingModels "model.safetensors") -Destination $PackagedMapAnythingModels -Force
$PackagedDa3Models = Join-Path $PackagedGeometryModels "da3nested-giant-large-1.1-noncommercial"
$null = New-Item -ItemType Directory -Path $PackagedDa3Models -Force
Copy-Item -LiteralPath (Join-Path $Da3Models "config.json") -Destination $PackagedDa3Models -Force
Copy-Item -LiteralPath (Join-Path $Da3Models "model.safetensors") -Destination $PackagedDa3Models -Force
if ($FlashInferReady -and (Test-Path -LiteralPath $FlashInferCacheArchive)) {
  Copy-Item -LiteralPath $FlashInferCacheArchive -Destination $PackagedGeometryModels -Force
}
# The frozen-worker validation must resolve only the assets beside the frozen
# executable. Do not let development paths conceal a packaging omission.
Remove-Item Env:SCANLAN_LINGBOT_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_LINGBOT_SKY_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_LINGBOT_DEPTH_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_MAPANYTHING_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_DA3_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_COLMAP_ALIKED_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_COLMAP_LIGHTGLUE_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:SCANLAN_FLASHINFER_CACHE_ARCHIVE -ErrorAction SilentlyContinue
& $PackagedWorker diagnostics `
  --require-cuda `
  --require-learned-features `
  --require-adaptive-frames
if ($LASTEXITCODE -ne 0) { throw "The packaged ScanLan splat runtime failed validation." }
& $PackagedGeometryWorker diagnostics `
  --require-lingbot `
  --require-lingbot-depth `
  --require-mapanything `
  --require-da3 `
  --require-flashinfer
if ($LASTEXITCODE -ne 0) { throw "The packaged ScanLan geometry runtime failed validation." }

Write-Host "Splat runtime ready: $PackagedWorker"
Write-Host "Geometry runtime ready: $PackagedGeometryWorker"
