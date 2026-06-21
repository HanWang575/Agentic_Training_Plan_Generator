param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$tmp = Join-Path $root "tmp"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

# Nested PowerShell sessions can contain both Path and PATH, which Start-Process rejects.
$currentPath = $env:Path
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", $currentPath, "Process")

if (-not (Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $python `
        -ArgumentList "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $tmp "program_api.out.log") `
        -RedirectStandardError (Join-Path $tmp "program_api.err.log") `
        -WindowStyle Hidden
}

if (-not (Get-NetTCPConnection -LocalPort $UiPort -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $python `
        -ArgumentList "-m", "streamlit", "run", "app\ui\streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "$UiPort", "--server.headless", "true" `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $tmp "program_ui.out.log") `
        -RedirectStandardError (Join-Path $tmp "program_ui.err.log") `
        -WindowStyle Hidden
}

Write-Output "API: http://127.0.0.1:$ApiPort"
Write-Output "UI: http://127.0.0.1:$UiPort"
