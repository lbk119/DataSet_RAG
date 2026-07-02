param(
    [switch]$SkipCheck
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $SkipCheck) {
    uv run python scripts/health_check.py
}

Write-Host "Starting import service on http://127.0.0.1:8001/import.html"
Start-Process powershell -WindowStyle Normal -ArgumentList "-NoExit", "-Command", "cd `"$Root`"; uv run python app/import_process/api/import_server.py"

Write-Host "Starting query service on http://127.0.0.1:8002/chat.html"
Start-Process powershell -WindowStyle Normal -ArgumentList "-NoExit", "-Command", "cd `"$Root`"; uv run python app/query_process/api/query_server.py"

Write-Host "Both service windows have been started."
