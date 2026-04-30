$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..."
py -3 -m venv venv

& .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Creating local directories..."
$dirs = @("data", "data\raw", "data\processed", "models", "reports", "logs")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from template. Edit it with your FOOTBALL_API_KEY."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate later with:  .\venv\Scripts\Activate.ps1"
Write-Host "Run pipeline with:    python run_pipeline.py"
