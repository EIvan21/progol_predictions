"""Per-concurso hit-rate history ("how many did we get right last week?").

Reads `progol_concurso_games` (predicted_label + actual_label, populated
by predict.py + settle_concurso_actuals) and renders a sortable table of
the last N concursos. Optional --csv writes a snapshot to reports/ so we
can chart trends across weekly trainer runs.

Note on historical coverage: concursos previous to the per-fixture
prediction history window have no `predicted_label` (we never persisted
the predictions made at the time). Those rows render as `—/14` rather
than a fake `0/14` — see render_table().

Usage:
    python -m src.progol.reporting.progol_history
    python -m src.progol.reporting.progol_history --n 12
    python -m src.progol.reporting.progol_history --csv reports/progol_history.csv

Mirrors the layout of reporting/league_dashboard.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

from src.progol import database


def summarize_recent(n: int = 8) -> List[dict]:
    """Returns up to `n` most-recent concurso summaries (newest first).
    Each item is the dict returned by database.get_concurso_hits, plus
    convenience fields used by the renderer:
        main_str / rev_str → 'h/c' formatted (or '-/t' when not yet comparable)
        status            → 'complete' | 'in_progress' | 'no_predictions'
    """
    out = []
    for cn in database.list_recent_concursos(n=n):
        s = database.get_concurso_hits(cn)
        m, r = s['main'], s['revancha']

        # Status label drives the "Notes" column. Order matters: a concurso
        # that has actuals but no predictions is "no_predictions" even though
        # technically nothing is pending.
        if m['predicted'] == 0 and r['predicted'] == 0:
            status = 'no_predictions'
        elif m['compared'] < m['predicted'] or r['compared'] < r['predicted']:
            status = 'in_progress'
        else:
            status = 'complete'

        s['main_str'] = _fmt_score(m)
        s['rev_str'] = _fmt_score(r)
        s['status'] = status
        out.append(s)
    return out


def _fmt_score(bucket: dict, dash: str = '-') -> str:
    """Display helper. Uses `compared` (slots where BOTH predicted and actual
    exist) as the denominator, NOT `settled` — otherwise concursos where we
    settled outcomes but never predicted would look like a 0/N whiff.
    States:
        - no compared slots (no predictions OR no outcomes yet)  → '-/14'
        - some compared, some still pending                      → '{h}/{c} of {t}'
        - everything compared                                    → '{h}/{t}'
    `dash` is parameterized so HTML callers can pass — while the CLI stays
    ASCII-safe for Windows cp1252 terminals."""
    total = bucket['total']
    compared = bucket['compared']
    hits = bucket['hits']
    if compared == 0:
        return f"{dash}/{total}"
    if compared < total:
        return f"{hits}/{compared} of {total}"
    return f"{hits}/{total}"


def render_table(rows: List[dict]) -> str:
    if not rows:
        return "(sin concursos en la base)"
    header = f"{'Concurso':>9}  {'Main':>14}  {'Revancha':>14}  {'Notas':<32}"
    sep = '-' * len(header)
    lines = [header, sep]
    main_hits = main_compared = 0
    rev_hits = rev_compared = 0
    n_concursos_with_compared = 0
    for s in rows:
        cn = s['concurso_number']
        m, r = s['main'], s['revancha']
        notes = _status_note(s)
        lines.append(
            f"{cn:>9}  {s['main_str']:>14}  {s['rev_str']:>14}  {notes:<32}"
        )
        if m['compared'] > 0 or r['compared'] > 0:
            main_hits += m['hits']
            main_compared += m['compared']
            rev_hits += r['hits']
            rev_compared += r['compared']
            n_concursos_with_compared += 1
    # Only emit PROMEDIO when we have at least one compared slot to average.
    if main_compared > 0 or rev_compared > 0:
        lines.append(sep)
        main_avg_str = (f"{main_hits}/{main_compared} ({main_hits/main_compared*100:.1f}%)"
                        if main_compared else "n/a")
        rev_avg_str = (f"{rev_hits}/{rev_compared} ({rev_hits/rev_compared*100:.1f}%)"
                       if rev_compared else "n/a")
        lines.append(
            f"{'PROMEDIO':>9}  main {main_avg_str}  rev {rev_avg_str}"
            f"   sobre {n_concursos_with_compared} concurso(s) con outcomes"
        )
    return "\n".join(lines)


def _status_note(s: dict) -> str:
    status = s['status']
    m, r = s['main'], s['revancha']
    if status == 'no_predictions':
        return "sin predicciones guardadas"
    if status == 'in_progress':
        pending_main = m['predicted'] - m['settled']
        pending_rev = r['predicted'] - r['settled']
        parts = []
        if pending_main:
            parts.append(f"{pending_main} main")
        if pending_rev:
            parts.append(f"{pending_rev} rev")
        return f"en curso ({' + '.join(parts)} pendientes)"
    return "cerrado"


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'concurso_number', 'status',
            'main_total', 'main_predicted', 'main_settled', 'main_compared', 'main_hits',
            'rev_total', 'rev_predicted', 'rev_settled', 'rev_compared', 'rev_hits',
        ])
        for s in rows:
            m, r = s['main'], s['revancha']
            w.writerow([
                s['concurso_number'], s['status'],
                m['total'], m['predicted'], m['settled'], m['compared'], m['hits'],
                r['total'], r['predicted'], r['settled'], r['compared'], r['hits'],
            ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=8,
                    help='Number of most-recent concursos to include (default: 8)')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Optional path to write CSV alongside the printed table')
    args = ap.parse_args(argv)

    rows = summarize_recent(n=args.n)
    if not rows:
        print("ERROR: no concursos in database", file=sys.stderr)
        return 1

    print(render_table(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
