param(
  [switch]$SkipPrepare,
  [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot "build"
$PortableName = "ScanLan-splat-portable"
$PortableRoot = Join-Path $BuildRoot $PortableName
$ArchivePath = Join-Path $BuildRoot "$PortableName.zip"

function Invoke-Checked([string]$Label, [scriptblock]$Command) {
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE."
  }
}

function Copy-Required([string]$Source, [string]$Destination) {
  if (-not (Test-Path -LiteralPath $Source)) {
    throw "Required portable resource is missing: $Source"
  }
  Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
}

Push-Location $ProjectRoot
try {
  if (-not $SkipPrepare) {
    Invoke-Checked "Splat runtime preparation" { npm.cmd run prepare:splat }
  }
  Invoke-Checked "Portable release build" {
    npm.cmd run tauri -- build --no-bundle --config src-tauri/tauri.splat.conf.json
  }

  $ResolvedBuildRoot = [System.IO.Path]::GetFullPath($BuildRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
  $ResolvedPortableRoot = [System.IO.Path]::GetFullPath($PortableRoot)
  if (-not $ResolvedPortableRoot.StartsWith("$ResolvedBuildRoot$([System.IO.Path]::DirectorySeparatorChar)", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Portable output escaped the expected build directory: $ResolvedPortableRoot"
  }
  if (Test-Path -LiteralPath $PortableRoot) {
    Remove-Item -LiteralPath $PortableRoot -Recurse -Force
  }
  $null = New-Item -ItemType Directory -Path $PortableRoot

  Copy-Required (Join-Path $ProjectRoot "src-tauri/target/release/scanlan.exe") (Join-Path $PortableRoot "ScanLan.exe")
  Copy-Required (Join-Path $ProjectRoot "worker/dist/scanlan-worker.exe") (Join-Path $PortableRoot "scanlan-worker.exe")
  Copy-Required (Join-Path $ProjectRoot "build/kinect-capture/Release") (Join-Path $PortableRoot "kinect2")
  Copy-Required (Join-Path $ProjectRoot "build/modern-capture/Release") (Join-Path $PortableRoot "modern")
  Copy-Required (Join-Path $ProjectRoot "splat-worker/dist/scanlan-splat") (Join-Path $PortableRoot "splat-runtime")
  Copy-Required (Join-Path $ProjectRoot "LICENSE") (Join-Path $PortableRoot "LICENSE")

  if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $ArchivePath) {
      Remove-Item -LiteralPath $ArchivePath -Force
    }
    Invoke-Checked "Portable ZIP creation" {
      tar.exe -a -cf $ArchivePath -C $BuildRoot $PortableName
    }
    Write-Host "Splat-enabled portable ZIP ready: $ArchivePath"
  } else {
    Write-Host "Splat-enabled portable directory ready: $PortableRoot"
  }
} finally {
  Pop-Location
}
