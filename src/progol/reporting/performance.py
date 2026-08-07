"""Model performance evaluation for the /rendimiento command and weekly tracking.

Actual results come ONLY from API-Football (the source of truth), fetched by
each fixture's `fixture_id` — never invented, never defaulted. Unfinished games
are reported as pending, not guessed. `settle_from_api` writes those results
into progol.db (`actual_label`) so /historial and retraining tracking see them;
`evaluate` reports how the model did per concurso vs simple baselines, split
into the 14 main games and the 7-game revancha, plus the expected number of
correct picks (the metric that maps to Ivan's 11-14 partial-prize objective).
"""
import json
import logging
import os

import requests

from src.progol import database

logger = logging.getLogger(__name__)

_API = "https://v3.football.api-sports.io"
_DONE = {"FT", "AET", "PEN"}
LBL = {0: "L", 1: "E", 2: "V"}


def _key():
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("FOOTBALL_API_KEY")


def _session():
    s = requests.Session()
    s.headers.update({"x-apisports-key": _key() or ""})
    return s


def _api_result(fixture_id, session, cache):
    """Return (label, 'gh-ga') from API-Football, or (None, status) if the
    match isn't finished / unavailable. label: 0=L, 1=E, 2=V. Never guesses."""
    if fixture_id in cache:
        return cache[fixture_id]
    try:
        r = session.get(f"{_API}/fixtures?id={fixture_id}", timeout=15).json()["response"][0]
        st = r["fixture"]["status"]["short"]
        gh, ga = r["goals"]["home"], r["goals"]["away"]
        if st not in _DONE or gh is None:
            out = (None, st)
        else:
            out = (0 if gh > ga else (1 if gh == ga else 2), f"{gh}-{ga}")
    except Exception as exc:
        logger.warning("api_result failed for %s: %s", fixture_id, exc)
        out = (None, "ERR")
    cache[fixture_id] = out
    return out


def _recent_concursos(n):
    conn = database.get_connection()
    rows = conn.execute(
        "SELECT DISTINCT concurso_number FROM progol_concurso_games "
        "WHERE predicted_label IS NOT NULL ORDER BY concurso_number DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _last_settled_concurso():
    """Most recent concurso that already has at least one settled result —
    i.e. the last one actually played, not the open one."""
    conn = database.get_connection()
    row = conn.execute(
        "SELECT concurso_number FROM progol_concurso_games WHERE actual_label IS NOT NULL "
        "ORDER BY concurso_number DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else None


def settle_from_api(n_concursos=8):
    """Backfill actual_label in progol.db from API-Football for finished games
    of the most recent concursos. Idempotent. Returns games newly settled."""
    key = _key()
    if not key:
        logger.warning("FOOTBALL_API_KEY missing; cannot settle from API")
        return 0
    session = _session()
    cache = {}
    conn = database.get_connection()
    pending = conn.execute(
        "SELECT concurso_number, game_number, fixture_id FROM progol_concurso_games "
        "WHERE actual_label IS NULL AND fixture_id IS NOT NULL "
        "AND concurso_number IN (SELECT DISTINCT concurso_number FROM progol_concurso_games "
        "ORDER BY concurso_number DESC LIMIT ?)", (n_concursos,)
    ).fetchall()
    settled = 0
    for cn, gn, fid in pending:
        lbl, _ = _api_result(fid, session, cache)
        if lbl is not None:
            conn.execute(
                "UPDATE progol_concurso_games SET actual_label=?, settled_at=datetime('now') "
                "WHERE concurso_number=? AND game_number=?", (lbl, cn, gn))
            settled += 1
    conn.commit()
    conn.close()
    logger.info("settle_from_api: %d games settled", settled)
    return settled


def _reason(pl, probs, actual):
    """Heuristic label for WHY a pick missed — grounded, not guessed."""
    if actual == 1 and pl != 1:
        return "empate no llamado"
    p = probs.get(LBL[pl], 0.0) if probs else 0.0
    if p >= 0.55:
        return "sorpresa (favorito cayó)"
    if probs and max(probs.values()) < 0.40:
        return "baja confianza (~volado)"
    return "fallo de pick"


def _eval_bucket(conn, concursos, session, cache, lo, hi):
    """Metrics for game slots [lo, hi] across the given concursos."""
    per = []
    conf = {(p, a): 0 for p in (0, 1, 2) for a in (0, 1, 2)}
    tot_hits = tot_comp = base_local = 0
    exp_total = 0.0
    for cn in concursos:
        games = conn.execute(
            "SELECT fixture_id, predicted_label, predicted_probs, actual_label "
            "FROM progol_concurso_games WHERE concurso_number=? AND game_number BETWEEN ? AND ? "
            "ORDER BY game_number", (cn, lo, hi)).fetchall()
        hits = comp = 0
        exp = 0.0
        for fid, pl, pp, al in games:
            if pl is None:
                continue
            if al is None and fid is not None:
                al, _ = _api_result(fid, session, cache)
            if al is None:
                continue
            comp += 1
            probs = json.loads(pp) if pp else None
            if probs:
                exp += probs.get(LBL[pl], 0.0)
            if pl == al:
                hits += 1
            if al == 0:
                base_local += 1
            conf[(pl, al)] += 1
        per.append({"concurso": cn, "hits": hits, "comp": comp, "expected": round(exp, 1)})
        tot_hits += hits; tot_comp += comp; exp_total += exp
    return {
        "per_concurso": per,
        "total_hits": tot_hits, "total_comp": tot_comp,
        "accuracy": tot_hits / tot_comp if tot_comp else 0.0,
        "baseline_local": base_local / tot_comp if tot_comp else 0.0,
        "expected_hits": exp_total,  # expected #correct across all comp games
        "confusion": conf,
        "draws_real": sum(conf[(p, 1)] for p in (0, 1, 2)),
        "draws_hit": conf[(1, 1)],
    }


def evaluate(n=6):
    """Compare predictions vs real results for the last `n` concursos, split
    into the 14 main games and the 7-game revancha."""
    settle_from_api(n_concursos=n)
    session = _session()
    cache = {}
    conn = database.get_connection()
    concursos = _recent_concursos(n)
    main = _eval_bucket(conn, concursos, session, cache, 1, 14)
    revancha = _eval_bucket(conn, concursos, session, cache, 15, 21)
    conn.close()
    return {"n_concursos": len(concursos), "concursos": concursos,
            "main": main, "revancha": revancha}


def render_telegram(n=6, max_breakdown=8):
    """Full /rendimiento message (HTML) — 14 main + revancha separate, vs
    baseline, expected hits, the draw problem, and the last played concurso
    per-game with the real scoreline. Results are from API-Football."""
    r = evaluate(n=n)
    out = [f"<b>📊 Rendimiento del modelo</b> (últimos {r['n_concursos']} concursos)"]
    for title, b in (("14 PRINCIPALES", r["main"]), ("REVANCHA (7)", r["revancha"])):
        if not b["total_comp"]:
            continue
        acc = b["accuracy"] * 100
        base = b["baseline_local"] * 100
        vs = "✅ arriba" if acc > base else "🔴 abajo"
        lines = [f"<b>{title}</b>",
                 f"Acierto: <b>{b['total_hits']}/{b['total_comp']} ({acc:.0f}%)</b>",
                 f"Baseline 'todo Local': {base:.0f}% → {vs}",
                 f"Aciertos esperados (calibración): {b['expected_hits']:.0f}",
                 f"Empates: {b['draws_hit']}/{b['draws_real']} acertados"
                 + (" ⚠️" if b["draws_real"] and not b["draws_hit"] else ""),
                 "por concurso: " + " · ".join(
                     f"{pc['concurso']} {pc['hits']}/{pc['comp']}"
                     for pc in b["per_concurso"] if pc["comp"])]
        out.append("<pre>" + "\n".join(lines) + "</pre>")

    cn, rows = last_concurso_breakdown()
    played = [x for x in rows if x["ok"] is not None]
    if played:
        bl = [f"<b>Último concurso jugado: {cn}</b>"]
        for x in played[:max_breakdown]:
            mark = "✅" if x["ok"] else "❌"
            extra = "" if x["ok"] else f" — {x.get('reason', '')}"
            bl.append(f"{mark} {x['game']}. {x['home'][:12]} vs {x['away'][:10]} "
                      f"[{x['pred']}→{x['actual']} {x['score']}]{extra}")
        misses = sum(1 for x in played if not x["ok"])
        bl.append(f"… {len(played)-misses}/{len(played)} aciertos en este concurso")
        out.append("<pre>" + "\n".join(bl) + "</pre>")
    out.append("<i>Resultados: API-Football (marcador real, no estimado)</i>")
    return "\n".join(out)


def last_concurso_breakdown():
    """Per-game right/wrong + reason for the last PLAYED concurso, with the real
    scoreline (proof) from API-Football. Covers all 21 games."""
    cn = _last_settled_concurso()
    if cn is None:
        return None, []
    session = _session()
    cache = {}
    conn = database.get_connection()
    games = conn.execute(
        "SELECT game_number, home_name, away_name, fixture_id, predicted_label, predicted_probs "
        "FROM progol_concurso_games WHERE concurso_number=? ORDER BY game_number", (cn,)).fetchall()
    conn.close()
    out = []
    for gn, ho, aw, fid, pl, pp in games:
        lbl, score = _api_result(fid, session, cache) if fid else (None, "NR")
        probs = json.loads(pp) if pp else None
        row = {"game": gn, "home": ho, "away": aw,
               "pred": LBL.get(pl, "-"), "actual": LBL.get(lbl, None),
               "score": score, "ok": (pl == lbl) if lbl is not None else None}
        if lbl is not None and pl != lbl:
            row["reason"] = _reason(pl, probs, lbl)
        out.append(row)
    return cn, out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settle_from_api(n_concursos=8)
    r = evaluate(n=6)
    m, rv = r["main"], r["revancha"]
    logger.info("rendimiento: main %d/%d (%.0f%%, base %.0f%%) | revancha %d/%d (%.0f%%)",
                m["total_hits"], m["total_comp"], m["accuracy"] * 100, m["baseline_local"] * 100,
                rv["total_hits"], rv["total_comp"], rv["accuracy"] * 100)
