$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location "$root\server"
if (-not (Test-Path "node_modules")) { Write-Host "Installing server deps..." -ForegroundColor Yellow; npm install }
Pop-Location

Push-Location "$root\client"
if (-not (Test-Path "node_modules")) { Write-Host "Installing client deps..." -ForegroundColor Yellow; npm install }
Pop-Location

Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$root\server'; Write-Host '=== Annotation Server ===' -ForegroundColor Green; node index.js`""
Start-Sleep -Seconds 2

Set-Location "$root\client"
npx vite --open
