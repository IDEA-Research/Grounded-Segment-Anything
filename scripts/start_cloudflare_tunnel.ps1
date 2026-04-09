param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelToken,
    [string]$CloudflaredPath = "C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
)

if (-not (Test-Path $CloudflaredPath)) {
    throw "cloudflared not found: $CloudflaredPath"
}

Write-Host "Starting cloudflared named tunnel with token ..."
& $CloudflaredPath tunnel --no-autoupdate run --token $TunnelToken
