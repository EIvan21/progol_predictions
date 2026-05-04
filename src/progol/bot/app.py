"""Long-polling Telegram bot for Progol predictions.

Lives on a separate e2-micro VM (free-tier) so it can answer commands 24/7
independent of the weekly trainer run. State (predictions/latest.json + the
SQLite DB) is read after a `gsutil rsync` from the same bucket the trainer
writes to — one-way replication, the bot never writes back to GCS, but it
DOES write to the local DB (bot_users / bot_messages / bot_threads). Local
DB writes are ephemeral on this node — the trainer rebuilds its own copy
on each weekly run.

Auth lives in the bot_users table:
- Anyone can hit /start or /whoami; they get registered as pending.
- All other commands require status='active' and role in (owner|admin|user).
- Admin commands (/usuarios, /aprobar, /bloquear) need role in (owner|admin).
Seed your own chat_id once via:
    python -m src.progol.bot.manage_users add YOUR_CHAT_ID --role owner

Run:
    TELEGRAM_BOT_TOKEN=...
    GCS_BUCKET=progol-data-storage
    python -m src.progol.bot.app
"""
import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from thefuzz import fuzz

from src.progol import config, database
from src.progol.bot.formatting import (
    format_concurso_message,
    format_users_list,
    format_match_prediction,
)
from src.progol.ingest.get_progol_ids import clean_name

load_dotenv()
logger = logging.getLogger(__name__)


HELP_TEXT = (
    "*Progol Bot*\n\n"
    "Comandos:\n"
    "/ultima\\_prediccion\\_progol — última predicción guardada\n"
    "/predecir\\_progol — predicción del concurso actual (sincroniza primero)\n"
    "/predecir\\_partido EQUIPO\\_A vs EQUIPO\\_B — predicción de un partido del slate\n"
    "/whoami — tu chat\\_id, rol y status\n"
    "/help — esta ayuda"
)

ADMIN_HELP = (
    "*Admin*\n"
    "/usuarios — lista de usuarios\n"
    "/aprobar CHAT\\_ID \\[role\\] — aprueba (default user)\n"
    "/bloquear CHAT\\_ID — bloquea\n"
)


def _gcs_sync():
    """One-way pull from GCS so the bot sees fresh predictions/DB. Uses gsutil
    because google-auth fails on this image's mTLS metadata server."""
    bucket = os.getenv('GCS_BUCKET', 'progol-data-storage')
    pairs = [
        (f"gs://{bucket}/predictions", str(config.PROJECT_ROOT / 'predictions')),
        (f"gs://{bucket}/db", str(config.DATA_DIR)),
    ]
    for src, dst in pairs:
        Path(dst).mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(
                ['gsutil', '-m', 'rsync', '-r', src, dst],
                capture_output=True, text=True, timeout=120,
            )
            logger.info(f"gcs_sync {src} rc={r.returncode}")
        except Exception as exc:
            logger.warning(f"gcs_sync {src} failed: {exc}")


def _load_latest():
    """Returns (concurso_number, header, games, latest_meta).
    latest_meta is the predictions/latest.json dict if present (gives us
    top_quinielas + model metadata) or {} if missing. Falls back to the
    DB's max(concurso_number) so the bot keeps working when latest.json
    hasn't been pushed to GCS yet."""
    latest_path = config.PROJECT_ROOT / 'predictions' / 'latest.json'
    latest = {}
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text())
        except Exception:
            latest = {}
    concurso = latest.get('concurso_number') or database.get_latest_concurso_number()
    if not concurso:
        return None, None, [], latest
    header, games = database.get_concurso_with_games(concurso)
    return concurso, header, games, latest


async def _record_inbound(update: Update):
    """Refresh user profile + log the inbound message. Called from every
    handler (public + guarded) so the audit log captures unauthorized hits."""
    chat = update.effective_chat
    user = update.effective_user
    if not chat:
        return
    try:
        database.bot_upsert_user(
            chat_id=chat.id,
            user_id=user.id if user else None,
            username=user.username if user else None,
            first_name=user.first_name if user else None,
            last_name=user.last_name if user else None,
        )
        text = update.message.text if update.message else None
        command = None
        if text and text.startswith('/'):
            command = text.split()[0].lstrip('/').split('@')[0]
        database.bot_log_message(chat.id, 'in', text, command=command)
    except Exception:
        logger.exception("record_inbound_failed")


def _guard(level='user'):
    """level: 'user' or 'admin'. Always records inbound first; rejects with a
    helpful message including chat_id if not authorized."""
    by_level = {
        'user': ('owner', 'admin', 'user'),
        'admin': ('owner', 'admin'),
    }
    allowed = by_level[level]

    def deco(handler):
        async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            await _record_inbound(update)
            chat_id = update.effective_chat.id if update.effective_chat else None
            if chat_id is None:
                return
            if not database.bot_is_authorized(chat_id, allowed_roles=allowed):
                msg = (
                    f"No estás autorizado para este comando.\n"
                    f"Tu chat\\_id: `{chat_id}`\n"
                    f"Pide acceso al owner."
                )
                if update.message:
                    try:
                        await update.message.reply_text(msg, parse_mode='Markdown')
                    except Exception:
                        pass
                return
            try:
                await handler(update, ctx)
            except Exception as exc:
                logger.exception(f"handler_error: {exc}")
                if update.message:
                    await update.message.reply_text("Error interno; revisa logs.")
        return wrapper
    return deco


# --- Public commands -------------------------------------------------------

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _record_inbound(update)
    chat_id = update.effective_chat.id
    user = database.bot_get_user(chat_id)
    if user and user['status'] == 'active':
        text = HELP_TEXT
        if user['role'] in ('owner', 'admin'):
            text += "\n\n" + ADMIN_HELP
        await update.message.reply_text(text, parse_mode='Markdown')
        return
    await update.message.reply_text(
        f"Hola — tu solicitud quedó registrada.\n"
        f"Tu chat\\_id: `{chat_id}`\n"
        f"El owner debe aprobarte para usar las predicciones.",
        parse_mode='Markdown',
    )


async def cmd_whoami(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _record_inbound(update)
    chat_id = update.effective_chat.id
    tg_user = update.effective_user
    user = database.bot_get_user(chat_id)
    role = user['role'] if user else 'unknown'
    status = user['status'] if user else 'unknown'
    user_id = (user.get('user_id') if user else None) or (tg_user.id if tg_user else None)
    await update.message.reply_text(
        f"chat\\_id: `{chat_id}`\n"
        f"user\\_id: `{user_id}`\n"
        f"role: `{role}`\n"
        f"status: `{status}`",
        parse_mode='Markdown',
    )


# --- User commands ---------------------------------------------------------

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = database.bot_get_user(chat_id)
    text = HELP_TEXT
    if user and user['role'] in ('owner', 'admin'):
        text += "\n\n" + ADMIN_HELP
    await update.message.reply_text(text, parse_mode='Markdown')


async def cmd_ultima(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    concurso, _header, games, latest = _load_latest()
    if not concurso or not games:
        await update.message.reply_text("No hay predicciones guardadas aún.")
        return
    await update.message.reply_text(
        format_concurso_message(concurso, games, latest=latest),
        parse_mode='Markdown',
    )


async def cmd_predecir_progol(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Sincronizando con GCS...")
    _gcs_sync()
    concurso, _header, games, latest = _load_latest()
    if not concurso or not games:
        await update.message.reply_text("No hay predicciones disponibles tras sync.")
        return
    await update.message.reply_text(
        format_concurso_message(concurso, games, latest=latest),
        parse_mode='Markdown',
    )


async def cmd_predecir_partido(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args_text = ' '.join(ctx.args) if ctx.args else ''
    if ' vs ' not in args_text.lower():
        await update.message.reply_text(
            "Uso: `/predecir_partido EQUIPO_A vs EQUIPO_B`",
            parse_mode='Markdown',
        )
        return
    a_in, _, b_in = args_text.partition(' vs ')
    a_in, b_in = a_in.strip(), b_in.strip()
    if not a_in or not b_in:
        await update.message.reply_text(
            "Uso: `/predecir_partido EQUIPO_A vs EQUIPO_B`",
            parse_mode='Markdown',
        )
        return

    concurso, _header, games, latest = _load_latest()
    if not concurso or not games:
        await update.message.reply_text("No hay un concurso activo en la base.")
        return

    a_clean = clean_name(a_in)
    b_clean = clean_name(b_in)
    best_score = 0
    best_game = None
    for g in games:
        s_h = fuzz.token_sort_ratio(a_clean, clean_name(g['home_name']))
        s_a = fuzz.token_sort_ratio(b_clean, clean_name(g['away_name']))
        score = (s_h + s_a) / 2
        if score > best_score:
            best_score = score
            best_game = g

    if not best_game or best_score < 70:
        await update.message.reply_text(
            f"No encontré ese partido en el concurso {concurso} "
            f"(mejor coincidencia: {best_score:.0f})."
        )
        return

    probs = None
    if best_game.get('predicted_probs'):
        try:
            probs = json.loads(best_game['predicted_probs'])
        except Exception:
            probs = None
    if probs is None:
        preds = latest.get('predictions', [])
        idx = best_game['game_number'] - 1
        if 0 <= idx < len(preds):
            p = preds[idx]
            probs = [p['L'], p['E'], p['V']]

    if not probs:
        await update.message.reply_text(
            f"Encontré *{best_game['home_name']} vs {best_game['away_name']}* "
            f"(juego {best_game['game_number']}) pero no hay predicción guardada.",
            parse_mode='Markdown',
        )
        return

    await update.message.reply_text(
        format_match_prediction(concurso, best_game, probs),
        parse_mode='Markdown',
    )


# --- Admin commands --------------------------------------------------------

async def cmd_usuarios(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    users = database.bot_list_users()
    await update.message.reply_text(
        format_users_list(users), parse_mode='Markdown'
    )


async def cmd_aprobar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Uso: `/aprobar CHAT_ID [role]`", parse_mode='Markdown'
        )
        return
    try:
        target = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("CHAT_ID inválido.")
        return
    role = ctx.args[1] if len(ctx.args) > 1 else 'user'
    if role not in ('owner', 'admin', 'user'):
        await update.message.reply_text("role inválido (owner|admin|user).")
        return
    n = database.bot_set_role(target, role=role, status='active')
    await update.message.reply_text(
        f"chat\\_id `{target}` -> `{role}`/active ({n} cambio).",
        parse_mode='Markdown',
    )


async def cmd_bloquear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Uso: `/bloquear CHAT_ID`", parse_mode='Markdown'
        )
        return
    try:
        target = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("CHAT_ID inválido.")
        return
    n = database.bot_set_role(target, role='blocked', status='blocked')
    await update.message.reply_text(
        f"chat\\_id `{target}` bloqueado ({n} cambio).",
        parse_mode='Markdown',
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")

    database.init_db()
    _gcs_sync()
    logger.info("starting bot (auth backed by bot_users table)")

    app = Application.builder().token(token).build()

    # Public — anyone can hit these to register or check status.
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('whoami', cmd_whoami))

    user_g = _guard('user')
    app.add_handler(CommandHandler('help', user_g(cmd_help)))
    app.add_handler(CommandHandler('ultima_prediccion_progol', user_g(cmd_ultima)))
    app.add_handler(CommandHandler('predecir_progol', user_g(cmd_predecir_progol)))
    app.add_handler(CommandHandler('predecir_partido', user_g(cmd_predecir_partido)))

    admin_g = _guard('admin')
    app.add_handler(CommandHandler('usuarios', admin_g(cmd_usuarios)))
    app.add_handler(CommandHandler('aprobar', admin_g(cmd_aprobar)))
    app.add_handler(CommandHandler('bloquear', admin_g(cmd_bloquear)))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
