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
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from thefuzz import fuzz

from src.progol import config, database
from src.progol.bot.formatting import (
    _decode_probs,
    format_budget_plan,
    format_concurso_message,
    format_progol_history,
    format_users_list,
    format_match_prediction,
    format_upcoming_match_prediction,
)
from src.progol.ingest.get_progol_ids import clean_name
from src.progol.modeling.quiniela import BASE_COST_MXN

load_dotenv()
logger = logging.getLogger(__name__)


HELP_TEXT = (
    "<b>🤖 Progol Bot</b>\n\n"
    "<b>Predicciones</b>\n"
    "/ultima_prediccion_progol — última predicción guardada\n"
    "/predecir_progol — predicción del concurso actual (sincroniza primero)\n"
    "/predecir_partido EQUIPO_A vs EQUIPO_B — predice un partido del concurso o cualquier fixture en los próximos 7 días\n"
    "\n<b>Análisis</b>\n"
    "/presupuesto — plan óptimo de dobles/triples para tu presupuesto\n"
    "/historial — hits/14 + revancha de los últimos 8 concursos\n"
    "/cancelar — aborta una conversación en curso\n"
    "\n<b>Cuenta</b>\n"
    "/whoami — tu chat_id, rol y status\n"
    "/help — esta ayuda"
)

ADMIN_HELP = (
    "<b>Admin</b>\n"
    "/usuarios — lista de usuarios\n"
    "/aprobar CHAT_ID [role] — aprueba (default user)\n"
    "/bloquear CHAT_ID — bloquea\n"
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
                    f"Tu chat_id: <code>{chat_id}</code>\n"
                    f"Pide acceso al owner."
                )
                if update.message:
                    try:
                        await update.message.reply_text(msg, parse_mode='HTML')
                    except Exception:
                        pass
                return
            try:
                # Return value matters for ConversationHandler entry points
                # (state ID); plain CommandHandlers return None and are
                # unaffected.
                return await handler(update, ctx)
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
        await update.message.reply_text(text, parse_mode='HTML')
        return
    await update.message.reply_text(
        f"Hola — tu solicitud quedó registrada.\n"
        f"Tu chat_id: <code>{chat_id}</code>\n"
        f"El owner debe aprobarte para usar las predicciones.",
        parse_mode='HTML',
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
        f"chat_id: <code>{chat_id}</code>\n"
        f"user_id: <code>{user_id}</code>\n"
        f"role: <code>{role}</code>\n"
        f"status: <code>{status}</code>",
        parse_mode='HTML',
    )


# --- User commands ---------------------------------------------------------

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = database.bot_get_user(chat_id)
    text = HELP_TEXT
    if user and user['role'] in ('owner', 'admin'):
        text += "\n\n" + ADMIN_HELP
    await update.message.reply_text(text, parse_mode='HTML')


async def cmd_ultima(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    concurso, _header, games, latest = _load_latest()
    if not concurso or not games:
        await update.message.reply_text("No hay predicciones guardadas aún.")
        return
    await update.message.reply_text(
        format_concurso_message(concurso, games, latest=latest),
        parse_mode='HTML',
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
        parse_mode='HTML',
    )


async def cmd_historial(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Last N concursos: hits/14 main + hits/7 revancha + per-concurso notes.
    Syncs GCS first so a fixture that just settled (and was picked up by the
    trainer's settle_concurso_actuals pass) is visible without waiting for
    the next /predecir_progol."""
    await update.message.reply_text("Sincronizando con GCS...")
    _gcs_sync()
    # Lazy import: keeps the formatting module import-clean for the slim bot
    # but avoids the formatting module needing to know about reporting.
    from src.progol.reporting.progol_history import summarize_recent
    n = 8
    if ctx.args:
        try:
            n = max(1, min(20, int(ctx.args[0])))
        except ValueError:
            pass
    rows = summarize_recent(n=n)
    await update.message.reply_text(
        format_progol_history(rows), parse_mode='HTML'
    )


async def cmd_predecir_partido(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args_text = ' '.join(ctx.args) if ctx.args else ''
    if ' vs ' not in args_text.lower():
        await update.message.reply_text(
            "Uso: <code>/predecir_partido EQUIPO_A vs EQUIPO_B</code>",
            parse_mode='HTML',
        )
        return
    a_in, _, b_in = args_text.partition(' vs ')
    a_in, b_in = a_in.strip(), b_in.strip()
    if not a_in or not b_in:
        await update.message.reply_text(
            "Uso: <code>/predecir_partido EQUIPO_A vs EQUIPO_B</code>",
            parse_mode='HTML',
        )
        return

    # 1) Try the active concurso slate first.
    concurso, _header, games, latest = _load_latest()
    a_clean = clean_name(a_in)
    b_clean = clean_name(b_in)

    if concurso and games:
        best_score = 0
        best_game = None
        for g in games:
            s_h = fuzz.token_sort_ratio(a_clean, clean_name(g['home_name']))
            s_a = fuzz.token_sort_ratio(b_clean, clean_name(g['away_name']))
            score = (s_h + s_a) / 2
            if score > best_score:
                best_score = score
                best_game = g
        if best_game and best_score >= 70:
            probs = _decode_probs(best_game.get('predicted_probs'))
            if probs is None:
                preds = latest.get('predictions', [])
                idx = best_game['game_number'] - 1
                if 0 <= idx < len(preds):
                    p = preds[idx]
                    probs = [p['L'], p['E'], p['V']]
            if probs:
                await update.message.reply_text(
                    format_match_prediction(concurso, best_game, probs),
                    parse_mode='HTML',
                )
                return

    # 2) Fallback: arbitrary upcoming fixture from match_predictions.
    a_id, a_score = database.resolve_team_id_by_name(a_in)
    b_id, b_score = database.resolve_team_id_by_name(b_in)
    if not a_id or not b_id:
        from src.progol.bot.formatting import _html_escape
        await update.message.reply_text(
            f"No encontré los equipos en la base.\n"
            f"<code>{_html_escape(a_in)}</code> → score {a_score}\n"
            f"<code>{_html_escape(b_in)}</code> → score {b_score}",
            parse_mode='HTML',
        )
        return
    pred = database.get_upcoming_match_prediction(a_id, b_id, days_ahead=7)
    if not pred:
        await update.message.reply_text(
            "No hay un partido próximo entre esos equipos en los siguientes 7 días."
        )
        return
    await update.message.reply_text(
        format_upcoming_match_prediction(pred), parse_mode='HTML'
    )


# --- /presupuesto conversation --------------------------------------------
# State id for AWAIT_BUDGET. The conversation is per-chat per-user; entry is
# `/presupuesto`, fallback `/cancelar`. Probs + slate are stashed in
# ctx.user_data so the state handler doesn't have to re-load.

AWAIT_BUDGET = 1
# Neutral 1X2 prior for fixtures that haven't resolved yet (e.g. not in the
# API window). Mirrors predict.DEFAULT_SLOT_PRIOR so /presupuesto can optimize
# a partial slate; unresolved games surface as triples (lowest confidence).
DEFAULT_SLOT_PRIOR = (0.45, 0.25, 0.30)


async def cmd_presupuesto(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    concurso, _header, games, _latest = _load_latest()
    if not concurso or not games:
        await update.message.reply_text("No hay concurso activo en la base.")
        return ConversationHandler.END

    main_games = sorted(games, key=lambda g: g['game_number'])[:14]
    probs_list = []
    unresolved = []
    for g in main_games:
        p = _decode_probs(g.get('predicted_probs'))
        if not p:
            # Fixture not in the API yet (e.g. a league still on break). Use the
            # neutral prior instead of bailing so the optimizer can still run.
            p = list(DEFAULT_SLOT_PRIOR)
            unresolved.append(g['game_number'])
        probs_list.append(p)
    if len(probs_list) < 14:
        await update.message.reply_text(
            f"El concurso {concurso} tiene solo {len(probs_list)} juegos cargados (esperaba 14)."
        )
        return ConversationHandler.END

    ctx.user_data['budget_concurso'] = concurso
    ctx.user_data['budget_probs'] = probs_list
    ctx.user_data['budget_games'] = main_games
    note = ""
    if unresolved:
        note = (f"\n⚠️ {len(unresolved)} juego(s) sin resolver "
                f"({', '.join(map(str, unresolved))}) usan prior neutral.")
    await update.message.reply_text(
        f"<b>Concurso {concurso}</b> cargado.{note}\n"
        f"¿Cuál es tu presupuesto en MXN? (mínimo {int(BASE_COST_MXN)})\n"
        f"Manda /cancelar para abortar.",
        parse_mode='HTML',
    )
    return AWAIT_BUDGET


async def receive_budget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _record_inbound(update)
    text = (update.message.text or '').strip().replace('$', '').replace(',', '')
    try:
        budget = float(text)
    except ValueError:
        await update.message.reply_text(
            "No entendí el monto. Manda solo un número (ej: 100). /cancelar para abortar."
        )
        return AWAIT_BUDGET
    if budget < BASE_COST_MXN:
        await update.message.reply_text(
            f"El presupuesto mínimo es ${BASE_COST_MXN:.0f} MXN."
        )
        return AWAIT_BUDGET

    probs_list = ctx.user_data.get('budget_probs') or []
    games = ctx.user_data.get('budget_games') or []
    concurso = ctx.user_data.get('budget_concurso')
    if not probs_list:
        await update.message.reply_text(
            "Sesión perdida; vuelve a iniciar /presupuesto."
        )
        return ConversationHandler.END

    import numpy as np
    from src.progol.modeling.quiniela import optimize_budget

    probs_arr = np.array(probs_list)
    plan = optimize_budget(probs_arr, budget=budget)
    msg = format_budget_plan(concurso, plan, games=games, budget_input=budget)
    await update.message.reply_text(msg, parse_mode='HTML')
    for k in ('budget_probs', 'budget_games', 'budget_concurso'):
        ctx.user_data.pop(k, None)
    return ConversationHandler.END


async def cmd_cancelar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    for k in ('budget_probs', 'budget_games', 'budget_concurso'):
        ctx.user_data.pop(k, None)
    await update.message.reply_text("Conversación cancelada.")
    return ConversationHandler.END


# --- Admin commands --------------------------------------------------------

async def cmd_usuarios(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    users = database.bot_list_users()
    await update.message.reply_text(
        format_users_list(users), parse_mode='HTML'
    )


async def cmd_aprobar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Uso: <code>/aprobar CHAT_ID [role]</code>", parse_mode='HTML'
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
        f"chat_id <code>{target}</code> → <code>{role}</code>/active ({n} cambio).",
        parse_mode='HTML',
    )


async def cmd_bloquear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Uso: <code>/bloquear CHAT_ID</code>", parse_mode='HTML'
        )
        return
    try:
        target = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("CHAT_ID inválido.")
        return
    n = database.bot_set_role(target, role='blocked', status='blocked')
    await update.message.reply_text(
        f"chat_id <code>{target}</code> bloqueado ({n} cambio).",
        parse_mode='HTML',
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")

    # Sync first (overwrites progol.db from GCS), THEN init both schemas.
    # Bot tables live in bot.db which the sync never touches, so the
    # owner row survives across restarts.
    _gcs_sync()
    database.init_db()
    database.init_bot_db()
    logger.info("starting bot (auth backed by bot.db)")

    app = Application.builder().token(token).build()

    # Public — anyone can hit these to register or check status.
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('whoami', cmd_whoami))

    user_g = _guard('user')
    app.add_handler(CommandHandler('help', user_g(cmd_help)))
    app.add_handler(CommandHandler('ultima_prediccion_progol', user_g(cmd_ultima)))
    app.add_handler(CommandHandler('predecir_progol', user_g(cmd_predecir_progol)))
    app.add_handler(CommandHandler('predecir_partido', user_g(cmd_predecir_partido)))
    app.add_handler(CommandHandler('historial', user_g(cmd_historial)))

    # /presupuesto conversation. Entry is guarded; the state handler doesn't
    # need its own guard since it's only reachable after entry succeeds.
    presupuesto_conv = ConversationHandler(
        entry_points=[CommandHandler('presupuesto', user_g(cmd_presupuesto))],
        states={
            AWAIT_BUDGET: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_budget),
            ],
        },
        fallbacks=[CommandHandler('cancelar', cmd_cancelar)],
        per_chat=True,
        per_user=True,
    )
    app.add_handler(presupuesto_conv)

    admin_g = _guard('admin')
    app.add_handler(CommandHandler('usuarios', admin_g(cmd_usuarios)))
    app.add_handler(CommandHandler('aprobar', admin_g(cmd_aprobar)))
    app.add_handler(CommandHandler('bloquear', admin_g(cmd_bloquear)))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
