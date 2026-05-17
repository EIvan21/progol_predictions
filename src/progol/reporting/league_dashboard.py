"""Per-league diagnostics dashboard.

Reads the `per_league` block from the most recent metrics.json (written
by train.py + modeling/metrics.py) and renders a sortable text table.
Optional --csv writes to a CSV so we can chart trends across weekly
trainer runs.

Usage:
    python -m src.progol.reporting.league_dashboard
    python -m src.progol.reporting.league_dashboard --sort ece
    python -m src.progol.reporting.league_dashboard --csv reports/league.csv

Lives in `reporting/` (not modeling/) because it's a consumer of training
artifacts, not part of the training pipeline itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

from src.progol import config


# Friendly labels for the league_ids we ingest. Anything not in this map
# prints as "League {id}". Keep in sync with fetch_data.LEAGUES and
# get_progol_ids.leagues — the dashboard is the most user-facing place
# where unmapped IDs would be visible.
LEAGUE_NAMES = {
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    11: "Copa Sudamericana",
    13: "Copa Libertadores",
    39: "Premier League",
    40: "Championship",
    45: "FA Cup",
    48: "EFL Cup",
    61: "Ligue 1",
    66: "Coupe de France",
    71: "Brazil Serie A",
    78: "Bundesliga",
    79: "Bundesliga 2",
    81: "DFB-Pokal",
    88: "Eredivisie",
    94: "Portugal",
    128: "Argentina",
    135: "Serie A",
    137: "Coppa Italia",
    140: "La Liga",
    141: "La Liga 2",
    143: "Copa del Rey",
    144: "Belgium",
    179: "Scottish Premiership",
    197: "Greek Super League",
    235: "Russian Premier League",
    253: "MLS",
    262: "Liga MX",
    263: "Liga MX Expansion",
}


def _league_name(lid: int) -> str:
    return LEAGUE_NAMES.get(int(lid), f"League {lid}")


def _load_per_league(path: Optional[Path] = None) -> dict:
    p = path or config.METRICS_PATH
    if not p.exists():
        raise FileNotFoundError(f"metrics.json not found at {p}; run training first")
    data = json.loads(p.read_text())
    pl = data.get('per_league') or {}
    if not pl:
        raise RuntimeError(
            f"metrics.json at {p} has no per_league block. Batch D introduced "
            "it — retrain (`python -m src.progol.modeling.train`) to populate."
        )
    # JSON keys are strings; cast back to int for cleaner sorting / lookup.
    return {int(k): v for k, v in pl.items()}


def render_table(per_league: dict, sort_by: str = 'n') -> str:
    """Returns a fixed-width text table sorted by `sort_by`. Reverse-sort
    for accuracy / log_loss / brier / ece — for those, "best on top" means
    descending for accuracy and ascending for the loss metrics."""
    ascending_keys = {'log_loss', 'brier_score', 'ece'}  # lower = better
    rows = sorted(
        per_league.items(),
        key=lambda kv: kv[1].get(sort_by, 0),
        reverse=(sort_by not in ascending_keys),
    )
    header = f"{'League':<28} {'id':>5} {'n':>5}  {'acc':>6} {'brier':>6} {'logloss':>8} {'ece':>7}"
    sep = '-' * len(header)
    lines = [header, sep]
    for lid, m in rows:
        lines.append(
            f"{_league_name(lid):<28} {lid:>5} {m['n']:>5}  "
            f"{m['accuracy']:>6.3f} {m['brier_score']:>6.3f} "
            f"{m['log_loss']:>8.3f} {m['ece']:>7.4f}"
        )
    return "\n".join(lines)


def write_csv(per_league: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['league_id', 'league_name', 'n', 'accuracy', 'brier_score', 'log_loss', 'ece'])
        for lid, m in sorted(per_league.items(), key=lambda kv: -kv[1]['n']):
            w.writerow([lid, _league_name(lid), m['n'],
                        m['accuracy'], m['brier_score'], m['log_loss'], m['ece']])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sort', default='n',
                    choices=('n', 'accuracy', 'brier_score', 'log_loss', 'ece'),
                    help='Column to sort by (default: n, descending)')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Optional path to write a CSV alongside the printed table')
    ap.add_argument('--metrics', type=Path, default=None,
                    help='Override metrics.json path (default: config.METRICS_PATH)')
    args = ap.parse_args(argv)

    try:
        pl = _load_per_league(args.metrics)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(render_table(pl, sort_by=args.sort))
    if args.csv:
        write_csv(pl, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
