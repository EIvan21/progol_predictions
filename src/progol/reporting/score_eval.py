"""Weekly tracking of the exact-score models. Runs a temporal holdout
(`score_model.evaluate`) and appends 1X2 accuracy + log-loss for both backends
to a CSV, so the models' performance is visible over time as they retrain on
accumulating results — the score-model analogue of progol_history.
"""
import argparse
import csv
import datetime
import logging

import pandas as pd

from src.progol import config
from src.progol.ingest.international_results import load_results
from src.progol.modeling import score_model as sm

logger = logging.getLogger(__name__)

HISTORY_CSV = config.REPORT_DIR / "score_model_history.csv"
_FIELDS = ["date", "test_since", "n_test", "blend_weight",
           "dc_acc", "dc_logloss", "xgb_acc", "xgb_logloss",
           "blend_acc", "blend_logloss"]


def run(df: pd.DataFrame = None, test_since: str = None, history_path=None) -> dict:
    if df is None:
        df = load_results()
    if test_since is None:
        # Rolling: score the last ~12 months of results.
        test_since = (df["date"].max() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    res = sm.evaluate(df, test_since=test_since)
    row = {
        "date": datetime.date.today().isoformat(),
        "test_since": test_since,
        "n_test": res["n_test"],
        "blend_weight": res["blend_weight"],
        "dc_acc": round(res["dc"]["accuracy"], 4),
        "dc_logloss": round(res["dc"]["log_loss"], 4),
        "xgb_acc": round(res["xgb"]["accuracy"], 4),
        "xgb_logloss": round(res["xgb"]["log_loss"], 4),
        "blend_acc": round(res["blend"]["accuracy"], 4),
        "blend_logloss": round(res["blend"]["log_loss"], 4),
    }
    path = history_path or HISTORY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    logger.info("score_model_eval n=%d DC ll=%.3f | XGB ll=%.3f | BLEND ll=%.3f",
                row["n_test"], row["dc_logloss"], row["xgb_logloss"], row["blend_logloss"])
    return row


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-since", help="YYYY-MM-DD holdout start (default: last 12 months)")
    args = ap.parse_args()
    r = run(test_since=args.test_since)
    print(f"DC  acc={r['dc_acc']} logloss={r['dc_logloss']}  |  "
          f"XGB acc={r['xgb_acc']} logloss={r['xgb_logloss']}  (n={r['n_test']})")
    print(f"appended to {HISTORY_CSV}")
