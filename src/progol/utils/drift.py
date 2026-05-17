"""Out-of-distribution detection between training feature stats and a new batch.

Saves per-feature mean/std/quantiles at training time. At inference, computes a
z-score per feature and a Kolmogorov-Smirnov p-value over a window of recent
predictions. Flags drift when too many features exceed thresholds.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_stats(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, dict]:
    out = {}
    for c in columns:
        s = df[c].astype(float).dropna()
        out[c] = {
            "mean": float(s.mean()),
            "std": float(s.std() or 1.0),
            "q05": float(s.quantile(0.05)),
            "q95": float(s.quantile(0.95)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_row(row: dict, stats: dict, z_threshold: float = 4.0) -> Dict[str, float]:
    """Returns {feature: z} for features that exceed z_threshold."""
    flags = {}
    for c, s in stats.items():
        if c not in row:
            continue
        try:
            v = float(row[c])
        except (TypeError, ValueError):
            continue
        z = abs(v - s["mean"]) / max(s["std"], 1e-9)
        if z > z_threshold:
            flags[c] = round(z, 2)
    return flags


def check_batch(df: pd.DataFrame, stats: dict, ks_threshold: float = 0.2) -> Dict[str, dict]:
    """KS-style drift on each feature; returns {feature: {ks: float, drift: bool}}."""
    from scipy.stats import ks_2samp
    out = {}
    for c, s in stats.items():
        if c not in df.columns:
            continue
        sample = df[c].astype(float).dropna()
        if len(sample) < 20:
            continue
        ref = np.random.normal(s["mean"], s["std"], size=max(200, len(sample)))
        try:
            ks_stat, p = ks_2samp(sample.values, ref)
        except Exception:
            continue
        out[c] = {"ks": float(ks_stat), "p": float(p), "drift": bool(ks_stat > ks_threshold)}
    return out


def record_run(history_path: Path, n_fixtures: int, n_drift_fixtures: int,
               keep_runs: int = 12) -> dict:
    """Append a drift summary record for one predict run. Keeps the last
    `keep_runs` entries (default 12 = ~3 months at weekly cadence). Returns
    the updated history dict for further inspection.

    Persistent file lets check_pressure() reason across runs without the
    trainer VM needing to retain state — each fresh VM boot picks up the
    file from GCS via the normal rsync."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = {'runs': []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = {'runs': []}

    from datetime import datetime, timezone
    history['runs'].append({
        'at': datetime.now(timezone.utc).isoformat(),
        'n_fixtures': int(n_fixtures),
        'n_drift_fixtures': int(n_drift_fixtures),
        'drift_rate': float(n_drift_fixtures) / max(n_fixtures, 1),
    })
    history['runs'] = history['runs'][-keep_runs:]
    history_path.write_text(json.dumps(history, indent=2))
    return history


def check_pressure(history: dict, window: int = 3, rate_threshold: float = 0.50,
                   alert_path: Optional[Path] = None) -> Optional[str]:
    """Returns an alert message if drift fired for >=`rate_threshold` of
    fixtures in EACH of the last `window` runs, else None. The threshold
    + window are deliberately conservative — a single drift run is normal
    noise (a Russian league round, a slate full of cup matches); sustained
    drift across 3 weeks means feature distributions have actually
    shifted and a retrain is warranted.

    When triggered AND `alert_path` is given, writes a marker file. The
    weekly Cloud Scheduler job already retrains, so this is just a
    surfacing mechanism for the user / dashboard — not an auto-trigger.
    """
    runs = history.get('runs') or []
    if len(runs) < window:
        return None
    recent = runs[-window:]
    if all(r['drift_rate'] >= rate_threshold for r in recent):
        msg = (
            f"DRIFT PRESSURE: drift_rate >= {rate_threshold:.0%} for "
            f"{window} consecutive runs (latest: "
            f"{recent[-1]['drift_rate']:.0%}). Consider re-tuning + retraining "
            "with a fresh fetch_data pass."
        )
        if alert_path is not None:
            alert_path.parent.mkdir(parents=True, exist_ok=True)
            alert_path.write_text(msg + "\n")
        return msg
    return None
