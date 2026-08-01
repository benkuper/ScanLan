$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DebugExecutable = Join-Path $ProjectRoot "src-tauri/target/debug/scanlan.exe"
$RunningSupervisor = Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq "node.exe" -and
    $_.CommandLine -like "*$ProjectRoot*tauri.js*dev*"
  } |
  Select-Object -First 1
$RunningDebugApps = @(Get-Process -Name "scanlan" -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -eq $DebugExecutable })
$Listeners = @(Get-NetTCPConnection -LocalPort 1420 -State Listen -ErrorAction SilentlyContinue)

if ($RunningSupervisor -and $RunningDebugApps.Count -gt 0 -and $Listeners.Count -gt 0) {
  Write-Host "ScanLan debug mode is already running (PID $($RunningSupervisor.ProcessId))."
  exit 0
}

if ($RunningSupervisor) {
  Write-Host "Stopping stale ScanLan debug supervisor (PID $($RunningSupervisor.ProcessId))."
  $ProcessTable = @(Get-CimInstance Win32_Process)
  $OwnedProcessIds = [System.Collections.Generic.HashSet[int]]::new()
  $null = $OwnedProcessIds.Add([int]$RunningSupervisor.ProcessId)
  $Changed = $true
  while ($Changed) {
    $Changed = $false
    foreach ($Process in $ProcessTable) {
      if ($OwnedProcessIds.Contains([int]$Process.ParentProcessId) -and
          -not $OwnedProcessIds.Contains([int]$Process.ProcessId)) {
        $null = $OwnedProcessIds.Add([int]$Process.ProcessId)
        $Changed = $true
      }
    }
  }
  $OwnedProcessIds |
    Where-Object { $_ -ne $PID } |
    Sort-Object -Descending |
    ForEach-Object { Stop-Process -Id $_ -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 500
  $Listeners = @(Get-NetTCPConnection -LocalPort 1420 -State Listen -ErrorAction SilentlyContinue)
}

if ($RunningDebugApps.Count -gt 0 -and $Listeners.Count -gt 0) {
  Write-Host "ScanLan debug mode is already running."
  exit 0
}

if ($RunningDebugApps.Count -gt 0) {
  Write-Host "Stopping $($RunningDebugApps.Count) stale ScanLan debug process(es)."
  $RunningDebugApps | Stop-Process
  $RunningDebugApps | Wait-Process -ErrorAction SilentlyContinue
}

foreach ($Listener in $Listeners) {
  $Process = Get-CimInstance Win32_Process -Filter "ProcessId = $($Listener.OwningProcess)"
  $IsProjectVite = $Process.Name -eq "node.exe" -and
    $Process.CommandLine -like "*$ProjectRoot*node_modules*vite*"
  if (-not $IsProjectVite) {
    throw "Port 1420 is used by another application (PID $($Listener.OwningProcess))."
  }
  Write-Host "Stopping stale ScanLan dev server (PID $($Listener.OwningProcess))."
  Stop-Process -Id $Listener.OwningProcess
  Wait-Process -Id $Listener.OwningProcess -ErrorAction SilentlyContinue
}

& npm.cmd run tauri -- dev
exit $LASTEXITCODE
