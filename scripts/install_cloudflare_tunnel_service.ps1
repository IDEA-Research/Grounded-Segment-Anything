param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelToken,
    [string]$CloudflaredPath = "C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
)

if (-not (Test-Path $CloudflaredPath)) {
    throw "cloudflared not found: $CloudflaredPath"
}

Write-Host "Installing cloudflared as a Windows service ..."
& $CloudflaredPath service install $TunnelToken
Write-Host "Service install command completed."
