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
# Bot auth/threads/messages live in a separate sqlite file so the bot VM's
# `gsutil rsync` from GCS (which overwrites progol.db with the trainer's
# version) cannot wipe bot user state.
BOT_DB_PATH = DATA_DIR / "bot.db"
TRAIN_CSV = PROCESSED_DATA_DIR / "final_train_data.csv"
PRIMARY_MODEL_PATH = MODEL_DIR / "calibrated_ensemble.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_STATS_PATH = MODEL_DIR / "feature_stats.json"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
LATEST_POINTER_PATH = MODEL_DIR / "latest.json"
PROGOL_IDS_PATH = DATA_DIR / "current_progol_ids.json"

# venue (stadium name) is ~99% redundant with home_team_id since each club
# plays home games at one stadium. Switching to home_id lets TargetEncoder
# learn cleaner per-club home-advantage patterns instead of duplicating the
# signal across (venue, home_id) interactions. Loses signal for neutral-site
# finals (UCL/cup finals) — accepted as a minor cost.
CAT_COLS = ['home_id', 'referee', 'league_id']
# API-Football league IDs that are knock-out cup competitions rather than
# regular league play. Used to derive the `is_cup` feature, which lets the
# model share signal across cup competitions (knock-out pressure, mixed-tier
# opponents, lineup rotation) instead of treating each cup as an isolated
# slice via league_id alone.
CUP_LEAGUE_IDS = {
    2,    # UEFA Champions League
    3,    # UEFA Europa League
    11,   # CONMEBOL Sudamericana
    13,   # CONMEBOL Libertadores
    45,   # FA Cup (England)
    48,   # EFL Cup / Carabao Cup (England)
    66,   # Coupe de France
    81,   # DFB-Pokal (Germany)
    137,  # Coppa Italia
    143,  # Copa del Rey (Spain)
}
# Market probs are NOT trained as features: only ~22 of 33k historical rows have
# odds, so they'd be effectively constant. They're applied as an inference-time
# blend instead — see predict.py + MODEL_MARKET_BLEND.
FEATURE_COLS = [
    'xg_diff', 'elo_diff', 'rank_gap', 'momentum_diff', 'h2h_diff', 'is_artificial',
    'gf_ewma_diff', 'ga_ewma_diff', 'sf_ewma_diff', 'sos_gf_diff',
    'rest_diff',
    # Draw-prone (averages, not diffs) — pushes E when both teams trend toward
    # low-scoring or frequent draws.
    'total_goals_avg', 'draw_rate_avg',
    # Knock-out cup flag — see CUP_LEAGUE_IDS above.
    'is_cup',
    # Injuries diff at fixture time (home - away). Filled by fetch_data
    # from API /injuries; 0 when API has no data for that date. Captures
    # "team is missing 3 starters" type signal that EWMA features can't.
    'injuries_diff',
    # Raw per-side EWMA scoring rates. Redundant with the *_diff features
    # for GBMs (LGB/CatBoost will pick whichever splits better), but the
    # Poisson DC base learner needs the absolute values to estimate lambdas
    # — diffs alone collapse Bayern 3-1 Stuttgart and Burnley 1-(-1) Spurs
    # to the same +2 signal. See modeling/poisson_dc.py.
    'home_gf_ewma', 'away_gf_ewma', 'home_ga_ewma', 'away_ga_ewma',
]
MODEL_MARKET_BLEND_DEFAULT = 0.6


def get_data_limit(total_count):
    if IS_LOCAL_TEST:
        return max(100, int(total_count * 0.1))
    return total_count
