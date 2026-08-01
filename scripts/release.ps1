$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseExecutable = Join-Path $ProjectRoot "src-tauri/target/release/scanlan.exe"
$RunningReleaseApp = Get-Process -Name "scanlan" -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -eq $ReleaseExecutable } |
  Select-Object -First 1

if ($RunningReleaseApp) {
  Write-Host "ScanLan release mode is already running (PID $($RunningReleaseApp.Id))."
  exit 0
}

& npm.cmd run tauri -- build --no-bundle
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if (-not (Test-Path $ReleaseExecutable)) {
  throw "Release build completed without producing $ReleaseExecutable."
}

Write-Host "Starting optimized ScanLan release build."
Start-Process -FilePath $ReleaseExecutable -WorkingDirectory (Split-Path -Parent $ReleaseExecutable)
