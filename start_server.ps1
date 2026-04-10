param(
    [string]$TunnelToken = $env:CLOUDFLARE_TUNNEL_TOKEN,
    [int]$Port = 7589,
    [string]$PublicHostname = "dachuang.acha.xx.kg",
    [string]$CloudflaredPath = "C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe",
    [string]$HfHome = "D:\hf-cache"
)

$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class ProjectJobObject
{
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public long Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateJobObject(IntPtr jobAttributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(
        IntPtr job,
        int infoType,
        IntPtr jobObjectInfo,
        uint jobObjectInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr handle);
}
"@

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv310-gpu\Scripts\python.exe"
$appPath = Join-Path $projectRoot "gradio_app.py"
$statePath = Join-Path $projectRoot ".server-processes.json"
$stdoutLog = Join-Path $projectRoot "gradio_app.stdout.log"
$stderrLog = Join-Path $projectRoot "gradio_app.stderr.log"
$tunnelOutLog = Join-Path $projectRoot "acha_tunnel.out.log"
$tunnelErrLog = Join-Path $projectRoot "acha_tunnel.err.log"
$hubCache = Join-Path $HfHome "hub"
$windowsPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$jobObjectHandle = [IntPtr]::Zero

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

function Stop-StaleStateProcesses {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    try {
        $state = Get-Content $Path -Raw | ConvertFrom-Json
        foreach ($pidValue in @($state.python_pid, $state.tunnel_pid)) {
            if ($pidValue) {
                Stop-ManagedProcess -ProcessId ([int]$pidValue)
            }
        }
    }
    catch {
        Write-Warning "Failed to clean stale process state: $($_.Exception.Message)"
    }
    finally {
        Remove-Item $Path -Force -ErrorAction SilentlyContinue
    }
}

function Wait-Until {
    param(
        [scriptblock]$Condition,
        [int]$TimeoutSeconds,
        [string]$Description
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) {
            return $true
        }

        Start-Sleep -Seconds 2
    }

    throw "Timed out while waiting for $Description."
}

function New-KillOnCloseJobObject {
    $jobName = "GroundedSegmentAnything-" + [Guid]::NewGuid().ToString()
    $handle = [ProjectJobObject]::CreateJobObject([IntPtr]::Zero, $jobName)
    if ($handle -eq [IntPtr]::Zero) {
        throw "Failed to create Windows job object."
    }

    $limits = New-Object ProjectJobObject+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    $limits.BasicLimitInformation.LimitFlags = 0x2000

    $length = [Runtime.InteropServices.Marshal]::SizeOf($limits)
    $pointer = [Runtime.InteropServices.Marshal]::AllocHGlobal($length)
    try {
        [Runtime.InteropServices.Marshal]::StructureToPtr($limits, $pointer, $false)
        $ok = [ProjectJobObject]::SetInformationJobObject($handle, 9, $pointer, [uint32]$length)
        if (-not $ok) {
            throw "Failed to configure Windows job object."
        }
    }
    finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
    }

    return $handle
}

function Add-ProcessToJobObject {
    param(
        [IntPtr]$Handle,
        [System.Diagnostics.Process]$Process
    )

    if ($Handle -eq [IntPtr]::Zero) {
        throw "Job object handle is not valid."
    }

    $ok = [ProjectJobObject]::AssignProcessToJobObject($Handle, $Process.Handle)
    if (-not $ok) {
        throw "Failed to assign process $($Process.Id) to the job object."
    }
}

function Show-RecentLogs {
    Write-Host ""
    Write-Host "Recent Gradio stdout:" -ForegroundColor Yellow
    if (Test-Path $stdoutLog) {
        Get-Content $stdoutLog -Tail 40
    }

    Write-Host ""
    Write-Host "Recent Gradio stderr:" -ForegroundColor Yellow
    if (Test-Path $stderrLog) {
        Get-Content $stderrLog -Tail 40
    }

    Write-Host ""
    Write-Host "Recent tunnel stderr:" -ForegroundColor Yellow
    if (Test-Path $tunnelErrLog) {
        Get-Content $tunnelErrLog -Tail 60
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path $appPath)) {
    throw "Application file not found: $appPath"
}

if (-not (Test-Path $CloudflaredPath)) {
    throw "cloudflared not found: $CloudflaredPath"
}

if ([string]::IsNullOrWhiteSpace($TunnelToken)) {
    throw "Missing tunnel token. Set CLOUDFLARE_TUNNEL_TOKEN or pass -TunnelToken."
}

Stop-StaleStateProcesses -Path $statePath

$existingPython = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -eq $pythonExe
}
if ($existingPython) {
    throw "A project Python process is already running. Use stop_server.ps1 first if you want a clean restart."
}

if ((Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded) {
    throw "Port $Port is already in use."
}

foreach ($logPath in @($stdoutLog, $stderrLog, $tunnelOutLog, $tunnelErrLog)) {
    if (Test-Path $logPath) {
        Remove-Item $logPath -Force
    }
}

$pythonCommand = @"
`$env:HF_HOME = '$HfHome'
`$env:HUGGINGFACE_HUB_CACHE = '$hubCache'
`$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Set-Location '$projectRoot'
& '$pythonExe' '-u' '$appPath' '--port' '$Port'
"@

$jobObjectHandle = New-KillOnCloseJobObject

$pythonProcess = Start-Process -FilePath $windowsPowerShell `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $pythonCommand `
    -WorkingDirectory $projectRoot `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog
Add-ProcessToJobObject -Handle $jobObjectHandle -Process $pythonProcess

$tunnelProcess = Start-Process -FilePath $CloudflaredPath `
    -ArgumentList "tunnel", "--protocol", "http2", "--no-autoupdate", "run", "--token", $TunnelToken `
    -WorkingDirectory $projectRoot `
    -PassThru `
    -RedirectStandardOutput $tunnelOutLog `
    -RedirectStandardError $tunnelErrLog
Add-ProcessToJobObject -Handle $jobObjectHandle -Process $tunnelProcess

$state = @{
    python_pid = $pythonProcess.Id
    tunnel_pid = $tunnelProcess.Id
    port = $Port
    public_hostname = $PublicHostname
    started_at = (Get-Date).ToString("s")
}
$state | ConvertTo-Json | Set-Content -Path $statePath -Encoding UTF8

$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -MessageData @{ state_path = $statePath } -Action {
    $path = $event.MessageData.state_path
    if (Test-Path $path) {
        try {
            $savedState = Get-Content $path -Raw | ConvertFrom-Json
            foreach ($pidValue in @($savedState.python_pid, $savedState.tunnel_pid)) {
                if ($pidValue) {
                    Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
                }
            }
        }
        catch {
        }
        finally {
            Remove-Item $path -Force -ErrorAction SilentlyContinue
        }
    }
} | Out-Null

try {
    Wait-Until -TimeoutSeconds 90 -Description "Gradio to answer on http://127.0.0.1:$Port" -Condition {
        try {
            (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/" -TimeoutSec 5).StatusCode -eq 200
        }
        catch {
            $false
        }
    }

    Wait-Until -TimeoutSeconds 60 -Description "Cloudflare tunnel registration" -Condition {
        if (-not (Test-Path $tunnelErrLog)) {
            return $false
        }

        (Get-Content $tunnelErrLog -Raw) -match "Registered tunnel connection"
    }

    Write-Host ""
    Write-Host "Services are up." -ForegroundColor Green
    Write-Host "Local URL : http://127.0.0.1:$Port"
    Write-Host "Public URL: https://$PublicHostname"
    Write-Host "Logs      : gradio_app.stdout.log / gradio_app.stderr.log / acha_tunnel.err.log"
    Write-Host ""
    Write-Host "Keep this window open while presenting. Press Ctrl+C to stop both processes." -ForegroundColor Cyan

    while ($true) {
        Start-Sleep -Seconds 5

        if (-not (Get-Process -Id $pythonProcess.Id -ErrorAction SilentlyContinue)) {
            throw "Gradio process exited unexpectedly."
        }

        if (-not (Get-Process -Id $tunnelProcess.Id -ErrorAction SilentlyContinue)) {
            throw "cloudflared process exited unexpectedly."
        }
    }
}
catch {
    Show-RecentLogs
    throw
}
finally {
    Unregister-Event -SourceIdentifier PowerShell.Exiting -ErrorAction SilentlyContinue
    Stop-ManagedProcess -ProcessId $pythonProcess.Id
    Stop-ManagedProcess -ProcessId $tunnelProcess.Id
    Remove-Item $statePath -Force -ErrorAction SilentlyContinue
    if ($jobObjectHandle -ne [IntPtr]::Zero) {
        [ProjectJobObject]::CloseHandle($jobObjectHandle) | Out-Null
    }
}
