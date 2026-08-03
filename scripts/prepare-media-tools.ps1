$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ToolsRoot = Join-Path $ProjectRoot "media-tools"
$DownloadRoot = Join-Path $env:TEMP "ScanLan-media-tools"

$Tools = @(
  @{
    Name = "ffmpeg"
    Version = "8.1.2"
    Uri = "https://github.com/GyanD/codexffmpeg/releases/download/8.1.2/ffmpeg-8.1.2-full_build.zip"
    Sha256 = "b8cdefab5f50590a076c27c2b56b0294a0e6154faded28ba1ba05ebc4f801f57"
    Executable = "bin/ffmpeg.exe"
  },
  @{
    Name = "colmap"
    Version = "4.1.1-cuda"
    Uri = "https://github.com/colmap/colmap/releases/download/4.1.1/colmap-x64-windows-cuda.zip"
    Sha256 = "b06064e7e4bd34f5b4ef71b442d3537d95d57c666dbec5a3b475902ccd832b9b"
    Executable = "bin/colmap.exe"
  }
)

New-Item -ItemType Directory -Force -Path $ToolsRoot, $DownloadRoot | Out-Null

function Test-InstalledTool {
  param([Parameter(Mandatory = $true)][hashtable]$Tool)

  $Destination = Join-Path $ToolsRoot $Tool.Name
  $ManifestPath = Join-Path $Destination ".scanlan-tool.json"
  $ExecutablePath = Join-Path $Destination $Tool.Executable
  if (-not (Test-Path -LiteralPath $ManifestPath) -or
      -not (Test-Path -LiteralPath $ExecutablePath)) {
    return $false
  }
  try {
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    return $Manifest.version -eq $Tool.Version -and $Manifest.sha256 -eq $Tool.Sha256
  } catch {
    return $false
  }
}

function Install-ToolArchive {
  param([Parameter(Mandatory = $true)][hashtable]$Tool)

  if (Test-InstalledTool $Tool) {
    Write-Host "$($Tool.Name) $($Tool.Version) is ready."
    return
  }

  $ArchivePath = Join-Path $DownloadRoot "$($Tool.Name)-$($Tool.Version).zip"
  $ExtractRoot = Join-Path $ToolsRoot ".$($Tool.Name)-extract-$PID"
  $StagingRoot = Join-Path $ToolsRoot ".$($Tool.Name)-install-$PID"
  $Destination = Join-Path $ToolsRoot $Tool.Name

  try {
    if (Test-Path -LiteralPath $ArchivePath) {
      $ExistingHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($ExistingHash -ne $Tool.Sha256) {
        Remove-Item -LiteralPath $ArchivePath -Force
      }
    }
    if (-not (Test-Path -LiteralPath $ArchivePath)) {
      Write-Host "Downloading $($Tool.Name) $($Tool.Version)..."
      & curl.exe --fail --location --retry 3 --output $ArchivePath $Tool.Uri
      if ($LASTEXITCODE -ne 0) { throw "$($Tool.Name) download failed." }
    }

    $ArchiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ArchiveHash -ne $Tool.Sha256) {
      throw "$($Tool.Name) archive checksum mismatch. Expected $($Tool.Sha256), received $ArchiveHash."
    }

    Remove-Item -LiteralPath $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $ExtractRoot, $StagingRoot | Out-Null
    Write-Host "Extracting $($Tool.Name)..."
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractRoot -Force

    $ExtractedItems = @(Get-ChildItem -LiteralPath $ExtractRoot -Force)
    $ContentRoot = if ($ExtractedItems.Count -eq 1 -and $ExtractedItems[0].PSIsContainer) {
      $ExtractedItems[0].FullName
    } else { $ExtractRoot }
    Get-ChildItem -LiteralPath $ContentRoot -Force |
      Move-Item -Destination $StagingRoot

    $StagedExecutable = Join-Path $StagingRoot $Tool.Executable
    if (-not (Test-Path -LiteralPath $StagedExecutable)) {
      throw "$($Tool.Name) archive did not contain $($Tool.Executable)."
    }
    $Manifest = @{
      name = $Tool.Name
      version = $Tool.Version
      sha256 = $Tool.Sha256
      source = $Tool.Uri
      installedAtUtc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json
    [IO.File]::WriteAllText(
      (Join-Path $StagingRoot ".scanlan-tool.json"),
      $Manifest + [Environment]::NewLine
    )

    if (Test-Path -LiteralPath $Destination) {
      Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Move-Item -LiteralPath $StagingRoot -Destination $Destination
    Write-Host "$($Tool.Name) $($Tool.Version) installed at $Destination"
  } finally {
    Remove-Item -LiteralPath $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

foreach ($Tool in $Tools) {
  Install-ToolArchive $Tool
}

$Ffmpeg = Join-Path $ToolsRoot "ffmpeg/bin/ffmpeg.exe"
$Colmap = Join-Path $ToolsRoot "colmap/bin/colmap.exe"
$FfmpegVersion = & $Ffmpeg -version
if ($LASTEXITCODE -ne 0) { throw "The downloaded FFmpeg runtime failed validation." }
$FfmpegVersion | Select-Object -First 1
$ColmapPath = Split-Path -Parent $Colmap
$PreviousPath = $env:PATH
try {
  $env:PATH = $ColmapPath + [IO.Path]::PathSeparator + $env:PATH
  $ColmapHelp = & $Colmap -h
  if ($LASTEXITCODE -ne 0) { throw "The downloaded CUDA COLMAP runtime failed validation." }
  $ColmapHelp | Select-Object -First 3
} finally {
  $env:PATH = $PreviousPath
}

Write-Host "Photo/video tools ready."
