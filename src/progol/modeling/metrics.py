"""Calibration / discrimination diagnostics beyond bare accuracy + log-loss.

ECE (Expected Calibration Error) answers a different question than log-loss:
"when the model says 70% confident, does it actually win 70% of the time?"
A model with great log-loss can still be miscalibrated — over-confident on
easy cases and pessimistic on hard ones in ways that cancel out in NLL.

Per-league breakdown surfaces "my Liga MX accuracy is 0.55 but my Russian
Premier League accuracy is 0.42" type issues that the global average buries.
Crucial when adding new leagues to the training set — you want to know if
the global Brier improvement is genuine lift or just a mix shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Multi-class ECE: bin predictions by their top-class confidence,
    compare bin avg confidence to bin avg correctness. 0 = perfect
    calibration (top-class probs match top-class accuracy).

    Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks".
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _multiclass_brier(y_true, y_prob) -> float:
    """Same definition as train.calculate_brier_score (mean over per-class
    Brier scores). Duplicated here so this module is self-contained."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    scores = []
    for i in range(n_classes):
        scores.append(brier_score_loss((y_true == i).astype(int), y_prob[:, i]))
    return float(np.mean(scores))


def per_league_breakdown(league_ids, y_true, y_prob,
                         min_samples: int = 30) -> dict:
    """Returns {league_id: {n, accuracy, brier, log_loss, ece}}.

    Skips leagues with fewer than `min_samples` predictions in the
    evaluation set — small-sample noise makes per-league metrics
    meaningless for tail leagues. The aggregate `accuracy/brier/log_loss`
    we already track applies to those rows.
    """
    league_ids = np.asarray(league_ids)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict = {}
    for lid in np.unique(league_ids):
        mask = league_ids == lid
        n = int(mask.sum())
        if n < min_samples:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        preds = yp.argmax(axis=1)
        acc = float((preds == yt).mean())
        brier = _multiclass_brier(yt, yp)
        # log_loss needs all 3 classes present in the slice. If a league
        # has zero draws in the holdout, sklearn complains — guard with
        # the labels arg which fills in absent classes.
        try:
            ll = float(log_loss(yt, yp, labels=[0, 1, 2]))
        except ValueError:
            ll = float('nan')
        ece = expected_calibration_error(yt, yp)
        out[int(lid)] = {
            'n': n,
            'accuracy': acc,
            'brier_score': brier,
            'log_loss': ll,
            'ece': ece,
        }
    return out
