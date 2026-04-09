$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$statePath = Join-Path $projectRoot ".server-processes.json"
$pythonExe = Join-Path $projectRoot ".venv310-gpu\Scripts\python.exe"

function Stop-ManagedProcess {
    param([int]$ProcessId)

    if (-not $ProcessId) {
        return
    }

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }

    try {
        Stop-Process -Id $ProcessId -ErrorAction Stop
        $process.WaitForExit(10000) | Out-Null
    }
    catch {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

if (Test-Path $statePath) {
    try {
        $state = Get-Content $statePath -Raw | ConvertFrom-Json
        foreach ($pidValue in @($state.python_pid, $state.tunnel_pid)) {
            if ($pidValue) {
                Stop-ManagedProcess -ProcessId ([int]$pidValue)
            }
        }
    }
    finally {
        Remove-Item $statePath -Force -ErrorAction SilentlyContinue
    }
}

$pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -eq $pythonExe
}

foreach ($process in $pythonProcesses) {
    Stop-ManagedProcess -ProcessId $process.Id
}

Write-Host "Project server processes stopped." -ForegroundColor Green
