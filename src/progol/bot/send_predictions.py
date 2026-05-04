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

import requests
from dotenv import load_dotenv

from src.progol import config, database
from src.progol.bot.formatting import format_concurso_message

load_dotenv()
logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in env")
        return 1

    latest_path = config.PROJECT_ROOT / 'predictions' / 'latest.json'
    latest = {}
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text())
        except Exception:
            latest = {}
    concurso = latest.get('concurso_number') or database.get_latest_concurso_number()
    if not concurso:
        logger.error("no concurso available — run predict + scrape first")
        return 2
    _header, games = database.get_concurso_with_games(concurso)
    if not games:
        logger.error(f"concurso {concurso} has no games in DB")
        return 2

    text = format_concurso_message(concurso, games, latest=latest)
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
