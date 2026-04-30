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
