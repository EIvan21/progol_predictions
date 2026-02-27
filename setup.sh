#!/bin/bash

# 1. Create Python Virtual Environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install requests pandas numpy xgboost scikit-learn google-cloud-storage python-dotenv

# 3. GCP Infrastructure Setup
# Assuming gcloud is already configured with credentials and project
echo "Creating GCS bucket: progol-data-storage..."
gsutil mb -l us-central1 gs://progol-data-storage/ || echo "Bucket might already exist."

# 4. Create local folder structure
echo "Creating local directories..."
mkdir -p data/raw data/processed models logs

# 5. Create a .env template for API keys
if [ ! -f .env ]; then
    echo "Creating .env template. Please update it with your API-Football Key!"
    echo "FOOTBALL_API_KEY=YOUR_API_KEY_HERE" > .env
    echo "GCP_PROJECT_ID=YOUR_PROJECT_ID_HERE" >> .env
fi

echo "Setup complete! Activate with: source venv/bin/activate"
