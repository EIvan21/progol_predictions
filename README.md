# ⚽ Progol Prediction Engine (IPI)

An autonomous machine learning pipeline designed to forecast results for the "Progol" betting game in Mexico. The system leverages high-fidelity match statistics, advanced feature engineering, and a stacked ensemble architecture to transform sports variance into actionable probabilities.

---

## 🚀 Key Features

- **Multi-Model Ensemble:** Combines XGBoost, CatBoost, Random Forest, and a Deep MLP Neural Network.
- **Strategic Ingestion:** Parallelized API fetcher with a 10x speed boost and once-per-day caching.
- **Differential Rivalry Engine:** Features are calculated as gaps (Home - Away) to eliminate standard bias.
- **Efficiency Metrics:** Includes Offensive Efficiency (Goals/Shots) and Pressure Index (Possession x Corners).
- **Automated ID Resolution:** Scrapes the current Progol slate and matches it to API IDs using fuzzy logic.
- **Dual Mode Support:** 
  - `LOCAL TEST`: 10% sample for fast logic validation.
  - `PRODUCTION`: 100% data processing on GCP Virtual Machines.

---

## 🏗️ Project Structure

```text
progol-engine/
├── run_pipeline.py     # 🚀 Master Orchestrator (Single Entry Point)
├── run_eda.py          # 📊 Independent Strategic Audit Generator
├── fetch_data.py       # Parallel Ingestion & Database Enrichment
├── preprocess.py       # Differential Feature Engineering
├── train_model.py      # Stacked Ensemble & Bayesian Tuning (Optuna)
├── get_progol_ids.py   # Web Scraping & Match ID Resolver
├── predict_progol.py   # Final Inference Engine
├── database.py         # SQLite Storage Layer
├── config.py           # Execution Mode & Strategy Settings
├── reports/            # Generated PDF Performance/EDA Reports
└── data/               # SQLite database and last-fetch logs
```

---

## ⚙️ Setup & Installation

### 1. Requirements
Ensure you have Python 3.9+ and an API Key from [api-football.com](https://www.api-football.com/).

### 2. Initial Setup
```bash
bash setup.sh
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
FOOTBALL_API_KEY=your_api_key_here
```

---

## 📖 Usage Guide

### 🚀 Running the Pipeline
The pipeline is fully interactive. Simply run:
```bash
python3 run_pipeline.py
```
1. Choose **Execution Mode** (Local vs. Prod).
2. Choose **Model Architecture** (Ensemble is recommended).
3. Choose **Strategy** (Contextual is recommended).

### 📊 Exploratory Data Analysis
To generate a Strategic Audit report with heatmaps and field-tilt analysis:
```bash
python3 run_eda.py
```

### 📥 Downloading Reports from GCP VM
If running on a Google Cloud VM, use SCP to download results:
```bash
gcloud compute scp --recurse [VM_NAME]:~/progol_predictions/reports/ ./
```

---

## 🔬 Methodology (Strategy 7)
The current "Elite Contextual" strategy uses a **Differential Gap** approach:
- **Offensive Signal:** `(roll_gf_home - roll_gf_away)`
- **Efficiency Signal:** `(roll_gf / roll_sh)_home - (roll_gf / roll_sh)_away`
- **Field Tilt:** `(poss * corners)_home - (poss * corners)_away`

**Ensemble Meta-Learner:** A Logistic Regression model weights the component models to maximize F1-Macro score and stabilize probability outputs.

---

## ⚖️ License
This project is for academic and private analytical purposes only. Betting involves risk; use predictions responsibly.
