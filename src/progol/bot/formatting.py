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
    Returns a [L, E, V] list of floats, or None on failure."""
    if not raw:
        return None
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(v, list) and len(v) == 3:
            return [float(x) for x in v]
    except Exception:
        return None
    return None


def format_concurso_message(concurso_number, games, latest=None,
                            model_version=None, generated_at=None):
    """Build the Telegram body for a concurso. Works from DB-only data
    (the trainer fills `predicted_probs` in `progol_concurso_games`); if
    `latest.json` is also available we surface model_version + top quinielas
    on top of the per-game lines."""
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

        label_idx = max(range(3), key=lambda k: probs[k])
        pred = ['L', 'E', 'V'][label_idx]
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

    top = (latest or {}).get('top_quinielas') or []
    if top:
        lines.append("*Top quinielas*")
        for q in top[:5]:
            quiniela = q.get('quiniela', [])
            if isinstance(quiniela, list):
                quiniela = ''.join(quiniela)
            prob = q.get('joint_prob') or q.get('prob') or 0
            lines.append(f"• `{quiniela}` _p={prob:.2e}_")

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


def format_match_prediction(concurso_number, game, probs):
    """One-game response for /predecir_partido."""
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
