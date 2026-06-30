"""Per-fixture exact-score report: a three-panel figure (score heatmap, 1X2
bar, top-10 scorelines) for each match in the current slate, saved as one PNG
per match plus a combined PDF, and optionally pushed to Telegram.

Reads the resolved slate matchups from predictions/latest.json, fits the
Dixon-Coles score model once, then renders a panel per fixture.
"""
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from src.progol import config
from src.progol.ingest.international_results import load_results
from src.progol.modeling import score_model as sm

logger = logging.getLogger(__name__)

C_HOME, C_DRAW, C_AWAY = "#1b7a3d", "#9e9e9e", "#c0392b"
# World Cup fixtures are at neutral venues; no home advantage by default.
NEUTRAL_DEFAULT = True


def _abbr(name: str) -> str:
    return name[:3].upper()


def _plot_panel(M, home, away, concurso, neutral):
    pH, pD, pA = sm.outcome_probs(M)
    top = sm.top_scores(M, 10)
    ah, aa = _abbr(home), _abbr(away)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    venue = "cancha neutral" if neutral else "local-visitante"
    fig.suptitle(f"Marcador probable: {home} vs {away}  |  Concurso {concurso}  ({venue})\n"
                 f"Modelo: Dixon-Coles", fontsize=13, fontweight="bold")

    ax = axes[0]
    import pandas as pd
    hd = pd.DataFrame(M[:6, :6] * 100,
                      index=[f"{ah} {i}" for i in range(6)],
                      columns=[f"{aa} {j}" for j in range(6)])
    sns.heatmap(hd, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Probabilidad (%)"})
    ax.set_title("Probabilidad por marcador (%)", fontweight="bold")
    ax.set_xlabel(f"Goles {away}"); ax.set_ylabel(f"Goles {home}")

    ax = axes[1]
    labs = [f"{home}\ngana", "Empate", f"{away}\ngana"]
    vals = [pH * 100, pD * 100, pA * 100]
    bars = ax.bar(labs, vals, color=[C_HOME, C_DRAW, C_AWAY], width=0.5,
                  edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("Probabilidad de resultado", fontweight="bold")
    ax.set_ylabel("Probabilidad (%)")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    cmap = {"home": C_HOME, "draw": C_DRAW, "away": C_AWAY}
    labels = [f"{ah} {s} {aa}" for s in top["score"]]
    colors = [cmap[r] for r in top["result"]]
    ax.barh(labels[::-1], top["prob"][::-1], color=colors[::-1])
    for k, v in enumerate(top["prob"][::-1]):
        ax.text(v + 0.1, k, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title("Top 10 marcadores más probables", fontweight="bold")
    ax.set_xlabel("Probabilidad (%)")
    ax.legend(handles=[mpatches.Patch(color=C_HOME, label=f"{home} gana"),
                       mpatches.Patch(color=C_DRAW, label="Empate"),
                       mpatches.Patch(color=C_AWAY, label=f"{away} gana")],
              loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _load_matchups():
    if not config.PROJECT_ROOT.joinpath("predictions", "latest.json").exists():
        raise FileNotFoundError("predictions/latest.json missing — run predict first")
    data = json.loads((config.PROJECT_ROOT / "predictions" / "latest.json").read_text())
    concurso = data.get("concurso_number", "?")
    matchups = []
    for p in data["predictions"]:
        if " vs " in p.get("match", ""):
            home, away = p["match"].split(" vs ", 1)
            matchups.append((home.strip(), away.strip()))
    return concurso, matchups


def _send_photo(token, chat_id, path, caption):
    import requests
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                          data={"chat_id": chat_id, "caption": caption},
                          files={"photo": f}, timeout=60)
    if not r.ok:
        logger.warning("telegram sendPhoto failed for %s: %s", path, r.text[:200])


def run(neutral: bool = NEUTRAL_DEFAULT, to_telegram: bool = False) -> list:
    concurso, matchups = _load_matchups()
    config.SCORE_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    model = sm.fit_dixon_coles(load_results())

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if to_telegram and not (token and chat_id):
        logger.warning("telegram requested but TELEGRAM_* env missing; skipping send")
        to_telegram = False

    saved = []
    seen = set()
    with PdfPages(config.SCORE_REPORT_PDF) as pdf:
        for home, away in matchups:
            # Revancha repeats fixtures; render each unique pairing once.
            if (home, away) in seen:
                continue
            seen.add((home, away))
            rh, ra = sm.resolve_team(home, model), sm.resolve_team(away, model)
            if rh is None or ra is None:
                logger.warning("skipping %s vs %s (not a national team in dataset)", home, away)
                continue
            M = sm.score_matrix(model, rh, ra, neutral=neutral)
            fig = _plot_panel(M, home, away, concurso, neutral)
            png = config.SCORE_MAPS_DIR / f"{home}_vs_{away}.png".replace(" ", "_")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            saved.append(png)
            if to_telegram:
                pH, pD, pA = sm.outcome_probs(M)
                _send_photo(token, chat_id, png,
                            f"{home} {pH*100:.0f}% | E {pD*100:.0f}% | {away} {pA*100:.0f}%")

    logger.info("wrote %d score maps -> %s + %s", len(saved), config.SCORE_MAPS_DIR, config.SCORE_REPORT_PDF)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true", help="also push each map to Telegram")
    ap.add_argument("--home-away", action="store_true", help="use home advantage (default: neutral)")
    args = ap.parse_args()
    paths = run(neutral=not args.home_away, to_telegram=args.telegram)
    print(f"Generated {len(paths)} score maps. PDF: {config.SCORE_REPORT_PDF}")
