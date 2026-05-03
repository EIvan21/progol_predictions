"""Posts the latest Progol predictions to a Telegram chat.

Usage (env required):
    TELEGRAM_BOT_TOKEN=...    # from @BotFather
    TELEGRAM_CHAT_ID=...      # /getUpdates after sending a message to the bot

    python -m src.progol.bot.send_predictions

The bot is a thin scaffold: read latest.json + the concurso row from SQLite,
format a message, POST to the Telegram Bot API. No webhook server, no polling.
"""
import json
import logging
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

from src.progol import config, database

load_dotenv()
logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _format_message(latest, header, games):
    """Build the Markdown message body. Telegram cap is 4096 chars; 21 games
    + header fits comfortably under 2k."""
    concurso = latest.get('concurso_number') or (header['concurso_number'] if header else '?')
    generated_at = latest.get('generated_at', '')
    model_v = latest.get('model_version', '?')
    preds = latest.get('predictions', [])

    by_idx = {g['game_number']: g for g in games}

    lines = [f"*Progol {concurso}* — model `{model_v}`"]
    if generated_at:
        try:
            ts = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            lines.append(f"_Generated {ts.strftime('%Y-%m-%d %H:%M UTC')}_")
        except ValueError:
            pass
    lines.append("")
    lines.append("```")
    lines.append(f"{'#':>2}  {'MATCH':<32} {'L':>5} {'E':>5} {'V':>5}  PRED")
    for i, p in enumerate(preds, start=1):
        match = p['match'][:32]
        pl = p['L'] * 100
        pe = p['E'] * 100
        pv = p['V'] * 100
        idx = max(range(3), key=lambda k: [p['L'], p['E'], p['V']][k])
        pred = ['L', 'E', 'V'][idx]
        actual = by_idx.get(i, {}).get('actual_label')
        if actual is not None:
            actual_letter = ['L', 'E', 'V'][actual]
            mark = '+' if actual == idx else 'x'
            pred = f"{pred} ({actual_letter}{mark})"
        lines.append(f"{i:>2}  {match:<32} {pl:>4.0f}% {pe:>4.0f}% {pv:>4.0f}%  {pred}")
    lines.append("```")

    top = latest.get('top_quinielas') or []
    if top:
        lines.append("")
        lines.append("*Top quinielas*")
        for q in top[:5]:
            quiniela = q.get('quiniela', [])
            if isinstance(quiniela, list):
                quiniela = ''.join(quiniela)
            prob = q.get('joint_prob') or q.get('prob') or 0
            lines.append(f"`{quiniela}` p={prob:.2e}")

    return "\n".join(lines)


def send():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in env")
        return 1

    latest_path = config.PROJECT_ROOT / 'predictions' / 'latest.json'
    if not latest_path.exists():
        logger.error(f"{latest_path} not found — run predict first")
        return 2

    latest = json.loads(latest_path.read_text())
    concurso = latest.get('concurso_number')
    header, games = (None, [])
    if concurso:
        header, games = database.get_concurso_with_games(concurso)

    text = _format_message(latest, header, games)
    res = requests.post(
        TELEGRAM_API.format(token=token),
        json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'},
        timeout=15,
    )
    if not res.ok:
        logger.error(f"telegram_send_failed status={res.status_code} body={res.text[:300]}")
        return 3
    logger.info("telegram_sent ok")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    sys.exit(send())
