import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IS_LOCAL_TEST = os.getenv('IS_LOCAL_TEST', 'False').lower() == 'true'
WEIGHT_STRATEGY = int(os.getenv('WEIGHT_STRATEGY', 3))
MODEL_TYPE = os.getenv('MODEL_TYPE', 'Ensemble')

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

DB_PATH = DATA_DIR / "progol.db"
PREDICTIONS_DB_PATH = DATA_DIR / "predictions.db"
TRAIN_CSV = PROCESSED_DATA_DIR / "final_train_data.csv"
PRIMARY_MODEL_PATH = MODEL_DIR / "calibrated_ensemble.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_STATS_PATH = MODEL_DIR / "feature_stats.json"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
LATEST_POINTER_PATH = MODEL_DIR / "latest.json"
PROGOL_IDS_PATH = DATA_DIR / "current_progol_ids.json"

CAT_COLS = ['venue', 'referee', 'league_id']
# Market probs are NOT trained as features: only ~22 of 33k historical rows have
# odds, so they'd be effectively constant. They're applied as an inference-time
# blend instead — see predict.py + MODEL_MARKET_BLEND.
FEATURE_COLS = [
    'xg_diff', 'elo_diff', 'rank_gap', 'momentum_diff', 'h2h_diff', 'is_artificial',
    'gf_ewma_diff', 'ga_ewma_diff', 'sf_ewma_diff', 'sos_gf_diff',
    'rest_diff',
]
MODEL_MARKET_BLEND_DEFAULT = 0.6


def get_data_limit(total_count):
    if IS_LOCAL_TEST:
        return max(100, int(total_count * 0.1))
    return total_count
