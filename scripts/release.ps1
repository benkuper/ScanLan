$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseExecutable = Join-Path $ProjectRoot "src-tauri/target/release/scanlan.exe"
$RunningReleaseApp = Get-Process -Name "scanlan" -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -eq $ReleaseExecutable } |
  Select-Object -First 1

if ($RunningReleaseApp) {
  Write-Host "Stopping the previous ScanLan release build (PID $($RunningReleaseApp.Id))."
  $RunningReleaseApp | Stop-Process
  $RunningReleaseApp | Wait-Process -ErrorAction SilentlyContinue
}

& npm.cmd run prepare:splat
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& npm.cmd run tauri -- build --no-bundle --config src-tauri/tauri.splat.conf.json
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if (-not (Test-Path $ReleaseExecutable)) {
  throw "Release build completed without producing $ReleaseExecutable."
}

Write-Host "Starting optimized ScanLan release build."
Start-Process -FilePath $ReleaseExecutable -WorkingDirectory (Split-Path -Parent $ReleaseExecutable)
