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

# Exact-score model (Dixon-Coles) inputs/outputs. National-team history is not
# in progol.db (only club leagues are fetched for the 1X2 model), so the score
# model fits on the public martj42/international_results dataset instead.
INTL_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
INTL_RESULTS_PATH = DATA_DIR / "international_results.csv"
SCORE_MAPS_DIR = REPORT_DIR / "score_maps"
SCORE_REPORT_PDF = REPORT_DIR / "score_report.pdf"

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
]
MODEL_MARKET_BLEND_DEFAULT = 0.6
# Per-league override of the model/market blend weight (model weight; market
# weight is 1-blend). Sharper markets => trust market more (lower blend); thin
# / opaque markets => trust model more (higher blend). Default 0.60 applies to
# leagues NOT listed here.
#   - Top-5 European leagues: very sharp lines from Pinnacle/Bet365; the
#     market beats most public models. Drop model weight to 0.50-0.55.
#   - Liga MX / Liga MX Expansion: less liquid; model gets more weight.
#   - Russian Premier League: opaque post-2022 sanctions; trust model.
#   - Cup competitions: lineups uncertain, market less informed => model.
MODEL_MARKET_BLEND_BY_LEAGUE = {
    # Top-5 European domestic leagues — sharp markets, less model weight.
    39:  0.50,  # Premier League
    140: 0.50,  # La Liga
    78:  0.50,  # Bundesliga
    135: 0.50,  # Serie A
    61:  0.50,  # Ligue 1
    # Second tier European — moderately sharp.
    40:  0.60,  # Championship
    79:  0.60,  # Bundesliga 2
    141: 0.60,  # La Liga 2
    # MLS / South America — medium liquidity, slightly more model weight.
    253: 0.65,  # MLS
    71:  0.65,  # Brazil Serie A
    128: 0.65,  # Argentina
    # Smaller domestic leagues — less efficient markets.
    94:  0.70,  # Portugal
    88:  0.70,  # Eredivisie
    144: 0.70,  # Belgium
    179: 0.70,  # Scottish Premiership
    197: 0.70,  # Greek Super League
    # Mexican leagues — model has data advantage given the Liga MX focus.
    262: 0.75,  # Liga MX
    263: 0.85,  # Liga MX Expansion (thin lines)
    # Russian PL — opaque post-sanctions, trust model.
    235: 0.85,  # Russian Premier League
    # Cups — lineups rotate, market is less informed than for league play.
    2:   0.65,  # UEFA Champions League
    3:   0.65,  # UEFA Europa League
    11:  0.75,  # Copa Sudamericana
    13:  0.70,  # Copa Libertadores
    45:  0.70,  # FA Cup
    48:  0.75,  # EFL Cup
    66:  0.75,  # Coupe de France
    81:  0.70,  # DFB-Pokal
    137: 0.70,  # Coppa Italia
    143: 0.70,  # Copa del Rey
}


def market_blend_for(league_id) -> float:
    """Return the model-weight to use when blending model probs with
    bookmaker market probs for fixtures in `league_id`. Falls back to
    MODEL_MARKET_BLEND_DEFAULT for any league not in the override map."""
    try:
        return MODEL_MARKET_BLEND_BY_LEAGUE.get(int(league_id), MODEL_MARKET_BLEND_DEFAULT)
    except (TypeError, ValueError):
        return MODEL_MARKET_BLEND_DEFAULT


def get_data_limit(total_count):
    if IS_LOCAL_TEST:
        return max(100, int(total_count * 0.1))
    return total_count
