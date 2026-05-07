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

    # Split games into main + revancha so each gets its own table.
    main = [g for g in games if g['game_number'] <= 14]
    rev = [g for g in games if g['game_number'] > 14]

    def _render_table(rows):
        # Header + data lines, all with consistent spacing for monospace.
        lines = [f"{'#':>2}  {'PARTIDO':<{MATCH_NAME_WIDTH}}  {'L':>2} {'E':>2} {'V':>2}  {'PICK':<4}"]
        for g in rows:
            gn = g['game_number']
            mtch = _short_match((g.get('home_name') or '?').upper(),
                                (g.get('away_name') or '?').upper())
            mtch = _html_escape(mtch).ljust(MATCH_NAME_WIDTH)
            probs = _decode_probs(g.get('predicted_probs'))
            if probs is None:
                lines.append(f"{gn:>2}  {mtch}   ·  ·  ·   —")
                continue
            pick, _idx = _pick_label(probs)
            actual = g.get('actual_label')
            mark = ''
            if actual is not None:
                actual_letter = ['L', 'E', 'V'][actual]
                mark = ' ✓' if pick.startswith(actual_letter) else f' ✗{actual_letter}'
            lines.append(
                f"{gn:>2}  {mtch}  "
                f"{int(probs[0]*100):>2} {int(probs[1]*100):>2} {int(probs[2]*100):>2}  "
                f"{pick:<4}{mark}"
            )
        return "<pre>" + "\n".join(lines) + "</pre>"

    if main:
        out.append(_render_table(main))
    if rev:
        out.append("")
        out.append("<b>— Revancha —</b>")
        out.append(_render_table(rev))

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
    """HTML body for /presupuesto output. Header + summary table, then a
    per-match line marking single/double/triple. `plan` is a
    quiniela.BudgetPlan; `games` (optional) labels each match by team
    names instead of just match index."""
    name_for = {}
    if games:
        for g in games:
            name_for[g['game_number']] = _short_match(
                (g.get('home_name') or '?').upper(),
                (g.get('away_name') or '?').upper(),
            )

    base_cost = (plan.cost / plan.n_tickets) if plan.n_tickets else 0
    summary = []
    if budget_input is not None:
        summary.append(f"Presupuesto:  ${budget_input:.0f} MXN")
    summary.append(f"Costo:        ${plan.cost:.0f} ({plan.n_tickets} boletos · ${base_cost:.0f} c/u)")
    summary.append(f"Cobertura:    {plan.coverage_prob*100:.4f}%")
    summary.append(
        f"Triples ({len(plan.triples)}): "
        + (", ".join(str(m+1) for m in sorted(plan.triples)) or "—")
    )
    summary.append(
        f"Dobles  ({len(plan.doubles)}): "
        + (", ".join(str(m+1) for m in sorted(plan.doubles)) or "—")
    )

    out = [
        f"<b>💰 Plan optimizado — Progol {concurso_number}</b>",
        "<pre>" + "\n".join(summary) + "</pre>",
        "<i>Probabilidad de acertar los 14 con esta combinación</i>",
        "",
        "<b>Por partido</b>",
    ]

    # Per-match table: kind tag, idx, name, played outcomes, probs.
    kind_tag = {'triple': 'T', 'double': 'D', 'single': ' '}
    rows = [f"{'':1}  {'#':>2}  {'PARTIDO':<{MATCH_NAME_WIDTH}}  {'JUEGA':<7}  {'L':>2} {'E':>2} {'V':>2}"]
    for s in plan.matches_summary:
        idx = s['match_index']
        played = "/".join(s['played'])
        p = s['probs']
        nm = name_for.get(idx, '')
        nm = _html_escape(nm).ljust(MATCH_NAME_WIDTH)
        rows.append(
            f"{kind_tag[s['kind']]:1}  {idx:>2}  {nm}  {played:<7}  "
            f"{int(p['L']*100):>2} {int(p['E']*100):>2} {int(p['V']*100):>2}"
        )
    out.append("<pre>" + "\n".join(rows) + "</pre>")
    out.append("<i>T = triple · D = doble · espacio = single pick</i>")
    return "\n".join(out)


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
