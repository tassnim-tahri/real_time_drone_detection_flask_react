# Opens inbound TCP port for Flask so phones on the same Wi-Fi can reach the laptop.
# Run PowerShell AS ADMINISTRATOR once (or whenever you change PORT).
#
# Usage (default port 8000):
#   cd detection_pipeline/scripts
#   .\allow_firewall_port.ps1
# Custom port:
#   .\allow_firewall_port.ps1 -Port 9000

param(
    [int]$Port = 8000
)

$ruleName = "Drone Flask detection port $Port"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Run this script as Administrator (right-click PowerShell -> Run as administrator)."
    exit 1
}

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Private, Public, Domain

Write-Host "OK: Allowed inbound TCP $Port (Private, Public, Domain profiles)."
Write-Host "If the phone still times out:"
Write-Host "  - Confirm phone and laptop are on the same Wi-Fi (not guest / AP isolation)."
Write-Host "  - Use the laptop IPv4 from ipconfig (e.g. 192.168.1.x), not localhost."
exit 0
