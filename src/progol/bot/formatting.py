"""Telegram message formatting for the Progol bot.

Produces HTML messages (parse_mode='HTML'). HTML is preferred over Markdown
because (a) only `< > &` need escaping, and (b) `<pre>` blocks render in
fixed-width font on mobile, which lets us align per-match data into a real
table.

User-supplied strings (team names, usernames) all pass through `_html_escape`.
"""
import json
from datetime import datetime


def _html_escape(s):
    if s is None:
        return ''
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


def _decode_probs(raw):
    """progol_concurso_games.predicted_probs is a JSON-encoded TEXT column.
    The trainer writes it as a dict `{"L":..., "E":..., "V":...}`, but we
    also accept a 3-list for forward-compat. Returns a [L, E, V] list of
    floats, or None on failure."""
    if not raw:
        return None
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(v, dict):
            return [float(v['L']), float(v['E']), float(v['V'])]
        if isinstance(v, list) and len(v) == 3:
            return [float(x) for x in v]
    except Exception:
        return None
    return None


TOSSUP_THRESHOLD = 0.05  # |P_top - P_2nd| < 5pp -> show both letters
MATCH_NAME_WIDTH = 22    # truncate "HOME vs AWAY" to this many chars in tables
JUEGA_WIDTH = 5          # widest possible value is "L/E/V" (triple)


def _pick_label(probs):
    """Return the display label for the model's top pick. When the gap
    between top-1 and top-2 is below TOSSUP_THRESHOLD, we surface both
    letters as `L/V` so the reader knows the modal pick is barely the
    favorite — exactly the kind of match that should be played as a
    double in /presupuesto."""
    order = sorted(range(3), key=lambda k: -probs[k])
    top = order[0]
    second = order[1]
    if probs[top] - probs[second] < TOSSUP_THRESHOLD:
        return f"{'LEV'[top]}/{'LEV'[second]}", top
    return 'LEV'[top], top


def _trunc(s, n):
    s = str(s)
    return (s[:n - 1] + '…') if len(s) > n else s


def _short_match(home, away, width=MATCH_NAME_WIDTH):
    """Compact 'HOME vs AWAY' that fits in `width` chars. We split the
    budget evenly between the two sides so a long home name doesn't push
    the away name off-screen."""
    home = (home or '?').strip()
    away = (away or '?').strip()
    sep = ' vs '
    half = (width - len(sep)) // 2
    return f"{_trunc(home, half)}{sep}{_trunc(away, half)}"


def _render_match_row(gn, home, away, probs, juega, mark=''):
    """Single row of the per-match table. Width budget (~40 chars):
        2 (#) + 2 + 22 (partido) + 2 + 2 (L) + 1 + 2 (E) + 1 + 2 (V) + 2 + 5 (JUEGA) = 41
    Stays inside Telegram mobile <pre> width."""
    mtch = _html_escape(_short_match((home or '?').upper(),
                                     (away or '?').upper())).ljust(MATCH_NAME_WIDTH)
    if probs is None:
        return f"{gn:>2}  {mtch}   ·  ·  ·  {'—':<{JUEGA_WIDTH}}{mark}"
    return (
        f"{gn:>2}  {mtch}  "
        f"{int(probs[0]*100):>2} {int(probs[1]*100):>2} {int(probs[2]*100):>2}  "
        f"{juega:<{JUEGA_WIDTH}}{mark}"
    )


def _table_header():
    return f"{'#':>2}  {'PARTIDO':<{MATCH_NAME_WIDTH}}  {'L':>2} {'E':>2} {'V':>2}  {'JUEGA':<{JUEGA_WIDTH}}"


def format_concurso_message(concurso_number, games, latest=None,
                            model_version=None, generated_at=None):
    """HTML body for the weekly concurso. Renders a fixed-width per-match
    table inside <pre>, then the budget plan summary if `latest.json`
    carries one."""
    if latest is None:
        latest = {}
    model_version = model_version or latest.get('model_version')
    generated_at = generated_at or latest.get('generated_at')

    head = [f"<b>🎯 Progol {concurso_number}</b>"]
    meta_bits = []
    if model_version:
        meta_bits.append(f"modelo <code>{_html_escape(model_version)}</code>")
    if generated_at:
        try:
            ts = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            meta_bits.append(ts.strftime('%Y-%m-%d %H:%M UTC'))
        except (ValueError, AttributeError):
            pass
    if meta_bits:
        head.append(f"<i>{' · '.join(meta_bits)}</i>")
    out = ["\n".join(head), ""]

    # If a budget plan is attached, use its per-match `played` lists for the
    # JUEGA column (L / L/V / L/E/V) so the table directly reflects what the
    # plan says to bet — no need to cross-reference "Triples: 7, 12" against
    # the rows. Plan only covers main 14; revancha keeps the toss-up marker.
    plan = (latest or {}).get('plan')
    plan_played = {}
    if plan:
        for s in plan.get('matches_summary') or []:
            plan_played[s['match_index']] = "/".join(s['played'])

    main = [g for g in games if g['game_number'] <= 14]
    rev = [g for g in games if g['game_number'] > 14]

    def _render_table(rows, use_plan=True):
        lines = [_table_header()]
        for g in rows:
            gn = g['game_number']
            probs = _decode_probs(g.get('predicted_probs'))
            if probs is None:
                lines.append(_render_match_row(gn, g.get('home_name'), g.get('away_name'), None, ''))
                continue
            if use_plan and gn in plan_played:
                juega = plan_played[gn]
            else:
                juega, _ = _pick_label(probs)
            actual = g.get('actual_label')
            mark = ''
            if actual is not None:
                actual_letter = ['L', 'E', 'V'][actual]
                mark = ' ✓' if actual_letter in juega.split('/') else f' ✗{actual_letter}'
            lines.append(_render_match_row(gn, g.get('home_name'), g.get('away_name'),
                                           probs, juega, mark=mark))
        return "<pre>" + "\n".join(lines) + "</pre>"

    if main:
        out.append(_render_table(main, use_plan=True))
    if rev:
        out.append("")
        out.append("<b>— Revancha —</b>")
        out.append(_render_table(rev, use_plan=False))

    plan = (latest or {}).get('plan')
    if plan:
        budget = plan.get('budget')
        cost = plan.get('cost', 0)
        cov = plan.get('coverage_prob', 0) * 100
        nt = plan.get('n_tickets', 0)
        doubles = plan.get('doubles') or []
        triples = plan.get('triples') or []
        plan_lines = []
        plan_lines.append(f"Costo:      ${cost:.0f}  ({nt} boletos)")
        plan_lines.append(f"Cobertura:  {cov:.4f}%")
        if triples:
            plan_lines.append(f"Triples:    {', '.join(str(m+1) for m in sorted(triples))}")
        if doubles:
            plan_lines.append(f"Dobles:     {', '.join(str(m+1) for m in sorted(doubles))}")
        out.append("")
        budget_disp = f" (${int(budget)} MXN base)" if budget is not None else ""
        out.append(f"<b>💰 Plan recomendado</b>{_html_escape(budget_disp)}")
        out.append("<pre>" + "\n".join(plan_lines) + "</pre>")
        out.append("<i>Usa /presupuesto para otro monto</i>")

    return "\n".join(out).rstrip()


def format_users_list(users):
    """HTML list for /usuarios. Compact bullets so the mobile reader
    doesn't have to fight the formatter."""
    if not users:
        return "<i>Sin usuarios registrados.</i>"
    lines = ["<b>👥 Usuarios registrados</b>", ""]
    for u in users:
        name = _html_escape(u.get('first_name') or '(sin nombre)')
        if u.get('username'):
            name += f" @{_html_escape(u['username'])}"
        role = u.get('role') or '?'
        status = u.get('status') or '?'
        role_disp = f"<b>{_html_escape(role)}</b>" if role in ('owner', 'admin') else _html_escape(role)
        lines.append(
            f"• <code>{u['chat_id']}</code> — {name} — {role_disp}/{_html_escape(status)}"
        )
    return "\n".join(lines)


def format_upcoming_match_prediction(prediction):
    """Single-fixture response for /predecir_partido (arbitrary upcoming
    fixture, not in current concurso). `prediction` is a match_predictions
    row as a dict."""
    h = _html_escape((prediction.get('home_name') or '?').upper())
    a = _html_escape((prediction.get('away_name') or '?').upper())
    probs = _decode_probs(prediction.get('predicted_probs'))
    if not probs:
        return f"<b>{h}</b> vs <b>{a}</b>\n<i>sin predicción</i>"
    pick, _idx = _pick_label(probs)
    date_str = prediction.get('date') or ''
    try:
        ts = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        date_str = ts.strftime('%a %Y-%m-%d %H:%M UTC')
    except (ValueError, AttributeError):
        pass
    model_v = prediction.get('model_version')
    meta_bits = [_html_escape(date_str)] if date_str else []
    if model_v:
        meta_bits.append(f"modelo <code>{_html_escape(model_v)}</code>")
    body = [
        f"<b>{h}</b> vs <b>{a}</b>",
        f"<i>{' · '.join(meta_bits)}</i>" if meta_bits else "",
        f"<pre>L  {int(probs[0]*100):>2}%   E  {int(probs[1]*100):>2}%   V  {int(probs[2]*100):>2}%</pre>",
        f"Predicción: <b>{pick}</b>",
    ]
    return "\n".join(b for b in body if b)


def format_budget_plan(concurso_number, plan, games=None, budget_input=None):
    """HTML body for /presupuesto output. Same compact per-match table as
    format_concurso_message, with a header summary block on top so the
    user gets cost / coverage / counts at a glance.

    `plan` is a quiniela.BudgetPlan; `games` (optional) labels each match
    by team names. The JUEGA column directly shows what the plan plays
    per match (L, L/V, or L/E/V) — no need for a separate kind-tag column."""
    name_for = {}
    if games:
        for g in games:
            name_for[g['game_number']] = (g.get('home_name'), g.get('away_name'))

    base_cost = (plan.cost / plan.n_tickets) if plan.n_tickets else 0
    summary_lines = []
    if budget_input is not None:
        summary_lines.append(f"Presupuesto: ${budget_input:.0f} MXN")
    summary_lines.append(f"Costo:       ${plan.cost:.0f} ({plan.n_tickets} boletos · ${base_cost:.0f} c/u)")
    summary_lines.append(f"Cobertura:   {plan.coverage_prob*100:.4f}%")
    summary_lines.append(
        f"Triples ({len(plan.triples)}): "
        + (", ".join(str(m+1) for m in sorted(plan.triples)) or "—")
    )
    summary_lines.append(
        f"Dobles  ({len(plan.doubles)}): "
        + (", ".join(str(m+1) for m in sorted(plan.doubles)) or "—")
    )

    out = [
        f"<b>💰 Plan optimizado — Progol {concurso_number}</b>",
        "<pre>" + "\n".join(summary_lines) + "</pre>",
        "<i>Probabilidad de acertar los 14 con esta combinación</i>",
        "",
    ]

    rows = [_table_header()]
    for s in plan.matches_summary:
        idx = s['match_index']
        played = "/".join(s['played'])
        p = s['probs']
        home, away = name_for.get(idx, ('?', '?'))
        rows.append(_render_match_row(idx, home, away,
                                      [p['L'], p['E'], p['V']], played))
    out.append("<pre>" + "\n".join(rows) + "</pre>")
    return "\n".join(out)


def format_previous_concurso_recap(prev_concurso, hits):
    """One-line HTML recap "Concurso pasado #N: Main 8/14 ✅ · Rev 5/7" for
    appending to the weekly send_predictions message. Returns None when
    there's nothing comparable yet — caller should skip the append rather
    than show an empty line.

    `hits` is the dict returned by database.get_concurso_hits. We don't
    rebuild it here so callers can decide whether to query the DB at all.
    """
    if not hits:
        return None
    m = hits['main']
    r = hits['revancha']
    if m['compared'] == 0 and r['compared'] == 0:
        return None

    parts = []
    if m['compared'] > 0:
        rate = m['hits'] / m['compared']
        # Thresholds picked from the user-stated baseline: 8/14 is a "good
        # week" (57%) so ✅. 5/14 (36%) is fair, lower is rough. These are
        # display-only, never gate anything.
        if rate >= 0.55:
            emoji = '✅'
        elif rate >= 0.40:
            emoji = '➖'
        else:
            emoji = '❌'
        suffix = f" of {m['total']}" if m['compared'] < m['total'] else ''
        parts.append(f"Main {m['hits']}/{m['compared']}{suffix} {emoji}")
    if r['compared'] > 0:
        suffix = f" of {r['total']}" if r['compared'] < r['total'] else ''
        parts.append(f"Rev {r['hits']}/{r['compared']}{suffix}")
    return f"<i>📈 Concurso pasado #{prev_concurso}: {' · '.join(parts)}</i>"


def format_progol_history(rows):
    """HTML body for /historial. Compact table inside <pre> with the per-
    concurso hit count (main + revancha) for the last N concursos plus a
    PROMEDIO footer over the concursos that have at least one compared slot.
    Mirrors reporting/progol_history.render_table but uses an em-dash
    (HTML renders it fine) and Telegram-friendly widths."""
    if not rows:
        return "<i>Sin concursos en la base.</i>"

    DASH = '—'
    header_lines = ["<b>📊 Historial Progol</b>", ""]

    table_lines = [f"{'#':>5}  {'Main':>13}  {'Revancha':>13}  {'Notas'}"]
    table_lines.append("-" * 60)

    main_hits = main_compared = 0
    rev_hits = rev_compared = 0
    n_with_outcomes = 0

    for s in rows:
        cn = s['concurso_number']
        m, r = s['main'], s['revancha']
        main_str = _fmt_progol_score(m, dash=DASH)
        rev_str = _fmt_progol_score(r, dash=DASH)
        note = _progol_note(s)
        table_lines.append(
            f"{cn:>5}  {main_str:>13}  {rev_str:>13}  {note}"
        )
        if m['compared'] > 0 or r['compared'] > 0:
            main_hits += m['hits']
            main_compared += m['compared']
            rev_hits += r['hits']
            rev_compared += r['compared']
            n_with_outcomes += 1

    if main_compared > 0 or rev_compared > 0:
        table_lines.append("-" * 60)
        avg_main = (f"{main_hits}/{main_compared} ({main_hits/main_compared*100:.0f}%)"
                    if main_compared else "n/a")
        avg_rev = (f"{rev_hits}/{rev_compared} ({rev_hits/rev_compared*100:.0f}%)"
                   if rev_compared else "n/a")
        table_lines.append(
            f"AVG    main {avg_main}   rev {avg_rev}"
        )
        footer = (
            f"\n<i>Promedio sobre {n_with_outcomes} concurso(s) con resultados "
            "comparables. Concursos sin predicción persistida muestran "
            f"{DASH}/N.</i>"
        )
    else:
        footer = (
            "\n<i>Aún no hay concursos con predicción + resultado comparables. "
            "Las celdas vacías son normales hasta que cierren los fixtures.</i>"
        )

    return "\n".join(header_lines) + "<pre>" + "\n".join(table_lines) + "</pre>" + footer


def _fmt_progol_score(bucket, dash='—'):
    """Telegram-side mirror of reporting.progol_history._fmt_score. Kept
    here (not imported) to avoid pulling the reporting module + its CLI
    deps into the slim bot image."""
    total = bucket['total']
    compared = bucket['compared']
    hits = bucket['hits']
    if compared == 0:
        return f"{dash}/{total}"
    if compared < total:
        return f"{hits}/{compared} of {total}"
    return f"{hits}/{total}"


def _progol_note(s):
    status = s['status']
    m, r = s['main'], s['revancha']
    if status == 'no_predictions':
        return "sin predicción"
    if status == 'in_progress':
        pend_main = m['predicted'] - m['compared']
        pend_rev = r['predicted'] - r['compared']
        parts = []
        if pend_main:
            parts.append(f"{pend_main}m")
        if pend_rev:
            parts.append(f"{pend_rev}r")
        return f"en curso ({'+'.join(parts)} pend)" if parts else "en curso"
    return "cerrado"


def format_match_prediction(concurso_number, game, probs):
    """One-game response for /predecir_partido (active concurso)."""
    h = _html_escape((game.get('home_name') or '?').upper())
    a = _html_escape((game.get('away_name') or '?').upper())
    pick, _idx = _pick_label(probs)
    return (
        f"<b>Concurso {concurso_number}</b> · Juego {game['game_number']}\n"
        f"<b>{h}</b> vs <b>{a}</b>\n"
        f"<pre>L  {int(probs[0]*100):>2}%   E  {int(probs[1]*100):>2}%   V  {int(probs[2]*100):>2}%</pre>"
        f"Predicción: <b>{pick}</b>"
    )
