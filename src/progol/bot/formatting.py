"""Telegram message formatting for the Progol bot.

Produces legacy-Markdown messages (parse_mode='Markdown') that render with
real bold/italic on mobile rather than as fixed-width code blocks. All
user-supplied strings (team names, usernames) pass through `_md_escape`
so embedded `_` `*` `` ` `` `[` don't break formatting.
"""
import json
from datetime import datetime


def _md_escape(s):
    if s is None:
        return ''
    return (str(s)
            .replace('\\', '\\\\')
            .replace('_', '\\_')
            .replace('*', '\\*')
            .replace('`', '\\`')
            .replace('[', '\\['))


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


def format_concurso_message(concurso_number, games, latest=None,
                            model_version=None, generated_at=None):
    """Build the Telegram body for a concurso. Works from DB-only data
    (the trainer fills `predicted_probs` in `progol_concurso_games`); if
    `latest.json` is also available we surface model_version + the budget
    plan on top of the per-game lines."""
    if latest is None:
        latest = {}
    model_version = model_version or latest.get('model_version')
    generated_at = generated_at or latest.get('generated_at')

    header_bits = [f"*Progol {concurso_number}*"]
    meta_bits = []
    if model_version:
        meta_bits.append(f"modelo `{model_version}`")
    if generated_at:
        try:
            ts = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            meta_bits.append(ts.strftime('%Y-%m-%d %H:%M UTC'))
        except (ValueError, AttributeError):
            pass
    if meta_bits:
        header_bits.append("_" + " · ".join(meta_bits) + "_")
    lines = header_bits + [""]

    for g in games:
        gn = g['game_number']
        h = _md_escape((g.get('home_name') or '?').upper())[:24]
        a = _md_escape((g.get('away_name') or '?').upper())[:24]
        probs = _decode_probs(g.get('predicted_probs'))

        if probs is None:
            lines.append(f"*{gn}.* {h} vs {a}")
            lines.append("    _sin predicción_")
            lines.append("")
            continue

        pred, label_idx = _pick_label(probs)
        cells = [
            f"L {probs[0]*100:.0f}%",
            f"E {probs[1]*100:.0f}%",
            f"V {probs[2]*100:.0f}%",
        ]
        cells[label_idx] = f"*{cells[label_idx]}*"
        actual = g.get('actual_label')
        suffix = ""
        if actual is not None:
            actual_letter = ['L', 'E', 'V'][actual]
            mark = "✓" if actual == label_idx else "✗"
            suffix = f"  ({actual_letter} {mark})"
        lines.append(f"*{gn}.* {h} vs {a}")
        lines.append(f"    {' · '.join(cells)} → *{pred}*{suffix}")
        lines.append("")

        # Progol = 14 main games + 7 revancha. Drop a section divider so the
        # mobile reader can skim the main slate without paging through revancha.
        if gn == 14 and any(g2['game_number'] > 14 for g2 in games):
            lines.append("*— Revancha —*")
            lines.append("")

    plan = (latest or {}).get('plan')
    if plan:
        budget = plan.get('budget')
        cost = plan.get('cost', 0)
        cov = plan.get('coverage_prob', 0) * 100
        nt = plan.get('n_tickets', 0)
        doubles = plan.get('doubles') or []
        triples = plan.get('triples') or []
        lines.append("*Plan recomendado*")
        if budget is not None:
            lines.append(f"_Presupuesto base: ${budget:.0f} MXN_")
        lines.append(f"Costo: *${cost:.2f}* MXN — {nt} boletos")
        lines.append(f"Cobertura: *{cov:.4f}%*")
        if triples:
            lines.append(f"Triples: {', '.join(str(m+1) for m in sorted(triples))}")
        if doubles:
            lines.append(f"Dobles: {', '.join(str(m+1) for m in sorted(doubles))}")
        lines.append("_(usa /presupuesto para otro monto)_")

    return "\n".join(lines).rstrip()


def format_users_list(users):
    """Bullet-style list for /usuarios. Avoids the fixed-width block that
    renders cramped on mobile."""
    if not users:
        return "_Sin usuarios registrados._"
    lines = ["*Usuarios registrados*", ""]
    for u in users:
        name = _md_escape(u.get('first_name') or '(sin nombre)')
        if u.get('username'):
            name += f" @{_md_escape(u['username'])}"
        role = u.get('role') or '?'
        status = u.get('status') or '?'
        role_disp = f"*{role}*" if role in ('owner', 'admin') else role
        lines.append(
            f"• `{u['chat_id']}` — {name} — {role_disp}/{status}"
        )
    return "\n".join(lines)


def format_upcoming_match_prediction(prediction):
    """One-line response for /predecir_partido fallback (arbitrary upcoming
    fixture, not in current concurso). `prediction` is a match_predictions
    row as a dict."""
    h = _md_escape((prediction.get('home_name') or '?').upper())
    a = _md_escape((prediction.get('away_name') or '?').upper())
    probs = _decode_probs(prediction.get('predicted_probs'))
    if not probs:
        return f"*{h}* vs *{a}*\n_sin predicción_"
    label_idx = max(range(3), key=lambda k: probs[k])
    pred = ['L', 'E', 'V'][label_idx]
    cells = [
        f"L {probs[0]*100:.0f}%",
        f"E {probs[1]*100:.0f}%",
        f"V {probs[2]*100:.0f}%",
    ]
    cells[label_idx] = f"*{cells[label_idx]}*"
    date_str = prediction.get('date') or ''
    try:
        ts = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        date_str = ts.strftime('%a %Y-%m-%d %H:%M UTC')
    except (ValueError, AttributeError):
        pass
    model_v = prediction.get('model_version')
    meta_line = f"_{date_str}"
    if model_v:
        meta_line += f" · modelo `{model_v}`"
    meta_line += "_"
    return (
        f"*{h}* vs *{a}*\n"
        f"{meta_line}\n"
        f"{' · '.join(cells)}\n"
        f"Predicción: *{pred}*"
    )


def format_budget_plan(concurso_number, plan, games=None, budget_input=None):
    """Markdown body for /presupuesto. Shows costo / cobertura / triples /
    dobles up top, then a per-match line. `plan` is a quiniela.BudgetPlan;
    `games` (optional) is the concurso's game rows so we can label each match
    by team names instead of just match index."""
    name_for = {}
    if games:
        for g in games:
            name_for[g['game_number']] = (
                f"{(g.get('home_name') or '?').upper()} vs "
                f"{(g.get('away_name') or '?').upper()}"
            )

    base_cost = (plan.cost / plan.n_tickets) if plan.n_tickets else 0
    lines = [f"*Plan optimizado — Progol {concurso_number}*", ""]
    if budget_input is not None:
        lines.append(f"_Presupuesto solicitado: ${budget_input:.2f} MXN_")
    lines += [
        f"Costo: *${plan.cost:.2f}* MXN — {plan.n_tickets} boletos "
        f"a ${base_cost:.0f} c/u",
        f"Cobertura: *{plan.coverage_prob*100:.4f}%*",
        "_(prob. estimada de acertar los 14 juegos con esta combinación)_",
        f"Triples ({len(plan.triples)}): "
        + (", ".join(str(m+1) for m in sorted(plan.triples)) or "—"),
        f"Dobles ({len(plan.doubles)}): "
        + (", ".join(str(m+1) for m in sorted(plan.doubles)) or "—"),
        "",
        "*Por partido:*",
    ]
    kind_tag = {'triple': '*\\[T\\]*', 'double': '*\\[D\\]*', 'single': '\\[ \\]'}
    for s in plan.matches_summary:
        idx = s['match_index']
        played = "/".join(s['played'])
        p = s['probs']
        nm = name_for.get(idx, '')
        nm_disp = (nm[:36] + '…') if len(nm) > 36 else nm
        nm_safe = _md_escape(nm_disp) if nm_disp else ''
        head = f"{kind_tag[s['kind']]} *{idx}.*"
        if nm_safe:
            head += f" {nm_safe}"
        lines.append(head)
        lines.append(
            f"    juega *{played}* · "
            f"L:{p['L']*100:.0f}% E:{p['E']*100:.0f}% V:{p['V']*100:.0f}%"
        )
    return "\n".join(lines)


def format_match_prediction(concurso_number, game, probs):
    """One-game response for /predecir_partido (active concurso)."""
    h = _md_escape((game.get('home_name') or '?').upper())
    a = _md_escape((game.get('away_name') or '?').upper())
    label_idx = max(range(3), key=lambda k: probs[k])
    pred = ['L', 'E', 'V'][label_idx]
    cells = [
        f"L {probs[0]*100:.0f}%",
        f"E {probs[1]*100:.0f}%",
        f"V {probs[2]*100:.0f}%",
    ]
    cells[label_idx] = f"*{cells[label_idx]}*"
    return (
        f"*Concurso {concurso_number}* — Juego {game['game_number']}\n"
        f"*{h}* vs *{a}*\n"
        f"{' · '.join(cells)}\n"
        f"Predicción: *{pred}*"
    )
