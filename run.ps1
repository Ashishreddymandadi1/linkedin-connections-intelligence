# Start backend (8010) + frontend (5182) for local development.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path "$root\backend\.venv")) {
    Write-Host "Creating backend venv..." -ForegroundColor Cyan
    & py -3.11 -m venv "$root\backend\.venv"
    & "$root\backend\.venv\Scripts\pip.exe" install -q -r "$root\backend\requirements.txt"
}
if (-not (Test-Path "$root\backend\.env")) {
    Copy-Item "$root\.env.example" "$root\backend\.env"
    Write-Warning "Created backend\.env from the example - add APIFY_API_TOKEN + ANTHROPIC_API_KEY."
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "Installing frontend deps..." -ForegroundColor Cyan
    Push-Location "$root\frontend"; npm install; Pop-Location
}

Write-Host "backend  -> http://localhost:8010/docs" -ForegroundColor Green
Write-Host "frontend -> http://localhost:5182" -ForegroundColor Green

$backend = Start-Process -PassThru -FilePath "$root\backend\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8010", "--reload" `
    -WorkingDirectory "$root\backend"
$frontend = Start-Process -PassThru -FilePath "npm.cmd" -ArgumentList "run", "dev" `
    -WorkingDirectory "$root\frontend"

Write-Host "`nPIDs: backend=$($backend.Id) frontend=$($frontend.Id). Ctrl+C to stop this script (processes keep running)." -ForegroundColor DarkGray
Wait-Process -Id $backend.Id, $frontend.Id
