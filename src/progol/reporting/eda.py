"""Consolidated EDA report.

Generates a single PDF (`reports/eda_<timestamp>.pdf`) with six sections:
  1. Dataset overview
  2. Target behavior (global + by league + over time)
  3. Feature quality (distributions, MI, correlation, VIF)
  4. Stability over time (per-feature drift)
  5. Market calibration (implied probs vs realized outcomes)
  6. Auto-recommendations (low-MI / high-VIF / class imbalance)

If a trained model exists at `models/latest.json`, an appendix prints feature
importance pulled from the actual estimator.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from sklearn.feature_selection import mutual_info_classif

from src.progol import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200

PLOTS_DIR = config.REPORT_DIR / "eda_plots"
LEAGUE_MAP = {
    262: "Liga MX", 39: "Premier League", 140: "La Liga", 135: "Serie A",
    78: "Bundesliga", 61: "Ligue 1", 253: "MLS", 71: "Brasileirao",
    128: "Argentina LPF", 88: "Eredivisie", 94: "Primeira Liga",
}
TARGET_LABELS = {0: "Home", 1: "Draw", 2: "Away"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_processed() -> pd.DataFrame | None:
    if not config.TRAIN_CSV.exists():
        logging.error(f"Missing {config.TRAIN_CSV}. Run preprocess first.")
        return None
    df = pd.read_csv(config.TRAIN_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["league_name"] = df["league_id"].map(LEAGUE_MAP).fillna(df["league_id"].astype(str))
    df["target_label"] = df["target"].map(TARGET_LABELS)
    return df


def _load_raw_summary() -> dict:
    if not config.DB_PATH.exists():
        return {}
    with sqlite3.connect(config.DB_PATH) as conn:
        n_total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        n_ft = conn.execute("SELECT COUNT(*) FROM matches WHERE status = 'FT'").fetchone()[0]
        leagues = conn.execute("SELECT COUNT(DISTINCT league_id) FROM matches").fetchone()[0]
        date_range = conn.execute("SELECT MIN(date), MAX(date) FROM matches").fetchone()
    return {
        "total_rows": n_total,
        "finished": n_ft,
        "leagues": leagues,
        "date_min": date_range[0],
        "date_max": date_range[1],
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_outcome_global(df: pd.DataFrame) -> Path:
    counts = df["target_label"].value_counts().reindex(["Home", "Draw", "Away"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#2c7fb8", "#bdbdbd", "#d95f0e"])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / total:.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Outcome distribution (global)")
    ax.set_ylabel("Matches")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save("01_outcome_global.png")


def _plot_outcome_by_league(df: pd.DataFrame) -> Path:
    pivot = (df.groupby(["league_name", "target_label"]).size()
             .unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0).reindex(columns=["Home", "Draw", "Away"])
    pivot = pivot.sort_values("Home", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(pivot))))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=["#2c7fb8", "#bdbdbd", "#d95f0e"])
    ax.set_title("Outcome share by league")
    ax.set_xlabel("Share")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=True)
    return _save("02_outcome_by_league.png")


def _plot_outcome_over_time(df: pd.DataFrame) -> Path:
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    pivot = (df.groupby(["year", "target_label"]).size().unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0).reindex(columns=["Home", "Draw", "Away"])
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(ax=ax, marker="o", color=["#2c7fb8", "#bdbdbd", "#d95f0e"])
    ax.set_title("Outcome share over time")
    ax.set_ylabel("Share")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)
    return _save("03_outcome_over_time.png")


def _plot_feature_distributions(df: pd.DataFrame, features: list[str]) -> Path:
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=40, color="#2c7fb8", alpha=0.7, edgecolor="white")
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature distributions", fontsize=12, y=1.0)
    return _save("04_feature_distributions.png")


def _plot_feature_by_outcome(df: pd.DataFrame, features: list[str]) -> Path:
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.7))
    axes = axes.flatten()
    palette = {"Home": "#2c7fb8", "Draw": "#bdbdbd", "Away": "#d95f0e"}
    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[[feat, "target_label"]].replace([np.inf, -np.inf], np.nan).dropna()
        sns.boxplot(data=data, x="target_label", y=feat, ax=ax,
                    order=["Home", "Draw", "Away"],
                    palette=palette, showfliers=False)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature behaviour by outcome (median ± IQR)", fontsize=12, y=1.0)
    return _save("05_feature_by_outcome.png")


def _plot_correlation(df: pd.DataFrame, features: list[str]) -> Path:
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson correlation between model features")
    return _save("06_correlation.png")


def _plot_mi_ranking(mi: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(mi))))
    mi.plot(kind="barh", ax=ax, color="#3182bd")
    ax.set_title("Mutual Information vs target (higher = more signal)")
    ax.set_xlabel("MI score")
    ax.invert_yaxis()
    return _save("07_mi_ranking.png")


def _plot_stability(df: pd.DataFrame, features: list[str]) -> Path:
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")[features].mean()
    yearly_z = (yearly - yearly.mean()) / yearly.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.3 * len(features))))
    sns.heatmap(yearly_z.T, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature mean by year (z-scored across years) — drift detector")
    return _save("08_stability.png")


def _plot_market_calibration(df: pd.DataFrame) -> Path:
    cols = ["prob_market_h", "prob_market_d", "prob_market_a"]
    if not all(c in df.columns for c in cols):
        return None
    sub = df.dropna(subset=cols + ["target"]).copy()
    sub["realized"] = sub["target"].map({0: "h", 1: "d", 2: "a"})
    fig, ax = plt.subplots(figsize=(7, 7))
    bins = np.linspace(0, 1, 11)
    for outcome, color, label in [("h", "#2c7fb8", "Home"),
                                  ("d", "#bdbdbd", "Draw"),
                                  ("a", "#d95f0e", "Away")]:
        col = f"prob_market_{outcome}"
        sub["bin"] = pd.cut(sub[col], bins=bins, include_lowest=True)
        agg = sub.groupby("bin").apply(
            lambda g: pd.Series({"mean_p": g[col].mean(),
                                 "freq": (g["realized"] == outcome).mean(),
                                 "n": len(g)})
        ).dropna()
        ax.plot(agg["mean_p"], agg["freq"], marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5,
            label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Implied probability (bookmaker)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Market calibration — bookmaker implied prob vs realized outcome")
    ax.legend(loc="lower right")
    return _save("09_market_calibration.png")


# ---------------------------------------------------------------------------
# Quantitative analyses
# ---------------------------------------------------------------------------

def _compute_mi(df: pd.DataFrame, features: list[str]) -> pd.Series:
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    y = df["target"]
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.Series(scores, index=features).sort_values(ascending=False)


def _compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).astype(float)
    rows = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = float("nan")
        rows.append({"feature": col, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def _model_feature_importance() -> pd.Series | None:
    pointer = config.LATEST_POINTER_PATH
    if not pointer.exists():
        return None
    try:
        info = json.loads(pointer.read_text())
        model_path = Path(info["path"]) / "calibrated_ensemble.pkl"
        if not model_path.exists():
            return None
        import joblib
        model = joblib.load(model_path)
        # Stacking + Calibration wraps things — try a few common locations.
        est = getattr(model, "estimator", model)
        final = getattr(est, "final_estimator_", None) or getattr(est, "final_estimator", None)
        if final is not None and hasattr(final, "coef_"):
            coefs = np.abs(final.coef_).mean(axis=0)
            return pd.Series(coefs, name="importance").sort_values(ascending=False)
    except Exception as exc:
        logging.warning(f"Could not load model importances: {exc}")
    return None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class _Pdf(FPDF):
    def header(self):
        self.set_fill_color(20, 33, 61)
        self.rect(0, 0, 210, 18, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 13)
        self.cell(0, 18, "Progol Predictions - EDA Report", ln=1, align="C")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


_UNICODE_MAP = str.maketrans({
    "—": "-",  # em dash
    "–": "-",  # en dash
    "−": "-",  # minus sign
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
    "→": "->",  # right arrow
    "←": "<-",  # left arrow
    "✓": "[ok]",  # check mark
    "✗": "[x]",  # ballot x
    "•": "*",  # bullet
})


def _safe(text: str) -> str:
    return str(text).translate(_UNICODE_MAP).encode("latin-1", "replace").decode("latin-1")


def _h1(pdf: _Pdf, text: str):
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(20, 33, 61)
    pdf.cell(0, 9, _safe(text), ln=1)
    pdf.set_text_color(0, 0, 0)


def _h2(pdf: _Pdf, text: str):
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _safe(text), ln=1)


def _para(pdf: _Pdf, text: str):
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, _safe(text))
    pdf.ln(1)


def _img(pdf: _Pdf, path: Path | None, w: int = 180):
    if path is None or not path.exists():
        return
    if pdf.get_y() + 80 > 270:
        pdf.add_page()
    pdf.image(str(path), x=(210 - w) / 2, w=w)
    pdf.ln(3)


def _table(pdf: _Pdf, headers: list[str], rows: list[list[str]], col_widths: list[int]):
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, _safe(h), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    for row in rows:
        for v, w in zip(row, col_widths):
            pdf.cell(w, 6, _safe(v), border=1)
        pdf.ln()
    pdf.ln(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report() -> Path | None:
    df = _load_processed()
    if df is None or df.empty:
        return None

    features = [c for c in config.FEATURE_COLS if c in df.columns]
    raw = _load_raw_summary()

    logging.info("Generating plots...")
    p_outcome_g = _plot_outcome_global(df)
    p_outcome_l = _plot_outcome_by_league(df)
    p_outcome_t = _plot_outcome_over_time(df)
    p_feat_dist = _plot_feature_distributions(df, features)
    p_feat_box = _plot_feature_by_outcome(df, features)
    p_corr = _plot_correlation(df, features)

    logging.info("Computing MI / VIF...")
    mi = _compute_mi(df, features)
    vif = _compute_vif(df, features)
    p_mi = _plot_mi_ranking(mi)
    p_stab = _plot_stability(df, features)
    p_calib = _plot_market_calibration(df)
    importances = _model_feature_importance()

    logging.info("Compiling PDF...")
    pdf = _Pdf()
    pdf.add_page()

    # ── 1. Dataset overview ────────────────────────────────────────────────
    _h1(pdf, "1. Dataset overview")
    overview = (
        f"Processed rows: {len(df):,}\n"
        f"Date range: {df['date'].min().date() if df['date'].notna().any() else '?'}"
        f"  ->  {df['date'].max().date() if df['date'].notna().any() else '?'}\n"
        f"Distinct leagues: {df['league_id'].nunique()}\n"
        f"Distinct teams (home set): {df.get('home_id', pd.Series()).nunique() or 'n/a'}\n"
        f"Model features in use: {len(features)}\n"
    )
    if raw:
        overview += (
            f"Raw matches in DB: {raw['total_rows']:,}  "
            f"(finished: {raw['finished']:,}, leagues: {raw['leagues']})\n"
        )
    _para(pdf, overview)

    nulls = df[features].isna().mean().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if len(nulls):
        _h2(pdf, "Missing data on model features")
        _table(pdf, ["Feature", "Null %"],
               [[f, f"{v:.1%}"] for f, v in nulls.items()],
               [70, 30])
    else:
        _para(pdf, "No nulls in model features.")

    # ── 2. Target behavior ─────────────────────────────────────────────────
    pdf.add_page()
    _h1(pdf, "2. Target behaviour")
    _para(pdf,
          "Class distribution globally, sliced by league, and tracked over time. "
          "Drift in draw frequency or home advantage is a leading indicator of "
          "regime change.")
    _img(pdf, p_outcome_g)
    _img(pdf, p_outcome_l)
    _img(pdf, p_outcome_t)

    # ── 3. Feature quality ────────────────────────────────────────────────
    pdf.add_page()
    _h1(pdf, "3. Feature quality")
    _para(pdf,
          "Mutual information measures non-linear signal towards the target. "
          "Boxplots show how each feature separates the three outcomes. "
          "Correlation flags redundancy.")
    _img(pdf, p_feat_dist)
    _img(pdf, p_feat_box)
    _img(pdf, p_mi)
    _img(pdf, p_corr)

    pdf.add_page()
    _h2(pdf, "VIF (variance inflation factor)")
    _para(pdf, "VIF > 5 suggests redundancy; VIF > 10 is severe multicollinearity.")
    _table(pdf, ["Feature", "VIF"],
           [[r["feature"], f"{r['VIF']:.2f}" if pd.notna(r['VIF']) else "n/a"]
            for _, r in vif.iterrows()],
           [80, 30])

    # ── 4. Stability ───────────────────────────────────────────────────────
    pdf.add_page()
    _h1(pdf, "4. Stability over time")
    _para(pdf,
          "Per-feature yearly mean, z-scored across years. Large warm/cool blocks "
          "indicate the feature shifted regime — retrain windows or adapt encoding.")
    _img(pdf, p_stab)

    # ── 5. Market calibration ──────────────────────────────────────────────
    if p_calib is not None:
        pdf.add_page()
        _h1(pdf, "5. Market calibration")
        _para(pdf,
              "Bookmaker implied probability (after no-vig normalisation) vs "
              "realised frequency. A point above the diagonal means the market "
              "underprices that outcome — opportunity. Below the diagonal means "
              "it is overpriced.")
        _img(pdf, p_calib)

    # ── 6. Auto-recommendations ────────────────────────────────────────────
    pdf.add_page()
    _h1(pdf, "6. Auto-recommendations")
    recs: list[str] = []

    low_mi = mi[mi < 0.001]
    if len(low_mi):
        recs.append(
            f"- Low information features (MI < 0.001): {', '.join(low_mi.index)}. "
            f"Consider removing or replacing.")
    high_vif = vif[vif["VIF"] > 10]["feature"].tolist()
    if high_vif:
        recs.append(
            f"- High multicollinearity (VIF > 10): {', '.join(high_vif)}. "
            f"Consider PCA / dropping one of each correlated pair.")

    counts = df["target_label"].value_counts(normalize=True)
    if "Draw" in counts and counts["Draw"] < 0.20:
        recs.append(
            f"- Draws underrepresented ({counts['Draw']:.1%}). "
            f"Consider class weights or focal loss when training.")
    if "Home" in counts and counts["Home"] > 0.50:
        recs.append(
            f"- Strong home bias ({counts['Home']:.1%}). "
            f"Verify it is encoded as a feature and not absorbed by the prior.")

    if not recs:
        recs.append("- No automated red flags detected. Features look healthy.")
    _para(pdf, "\n".join(recs))

    # ── Appendix: trained-model feature importance ─────────────────────────
    if importances is not None and len(importances) == len(features):
        pdf.add_page()
        _h1(pdf, "Appendix. Trained model feature importance")
        _para(pdf,
              "Coefficients of the meta-learner (logistic regression) over the "
              "stacked base predictions, mapped back to the original features.")
        imp = pd.Series(importances.values, index=features).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(imp))))
        imp.plot(kind="barh", ax=ax, color="#2c7fb8")
        ax.invert_yaxis()
        ax.set_title("Meta-learner |coefficients|")
        appendix_path = _save("10_model_importance.png")
        _img(pdf, appendix_path)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.REPORT_DIR / f"eda_{timestamp}.pdf"
    pdf.output(str(out_path))
    logging.info(f"EDA report written: {out_path}")

    # Persist the quantitative tables as CSV for downstream use.
    mi.to_csv(config.REPORT_DIR / "mutual_information.csv", header=["mi_score"])
    vif.to_csv(config.REPORT_DIR / "vif_analysis.csv", index=False)
    return out_path


def main():
    out = build_report()
    if out is None:
        raise SystemExit(1)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
