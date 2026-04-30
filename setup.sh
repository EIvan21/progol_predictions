#!/usr/bin/env bash
set -euo pipefail

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating local directories..."
mkdir -p data/raw data/processed models reports logs

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template. Edit it with your FOOTBALL_API_KEY."
fi

echo ""
echo "Setup complete."
echo "Activate later with:  source venv/bin/activate"
echo "Run pipeline with:    python run_pipeline.py"
