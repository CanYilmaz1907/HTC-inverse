from __future__ import annotations

import datetime as dt
from typing import Callable, Awaitable

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from bybit_client import BybitClient
from config import AppConfig
from scanner import ScanSummary
from zoneinfo import ZoneInfo


def build_application(config: AppConfig, client: BybitClient) -> Application:
    application = (
        ApplicationBuilder()
        .token(config.telegram.bot_token)
        .build()
    )

    # Store shared objects in bot_data
    application.bot_data["config"] = config
    application.bot_data["bybit_client"] = client

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("scan", scan))
    application.add_handler(CommandHandler("scan_rise", scan_rise))
    application.add_handler(CommandHandler("scan_fall", scan_fall))

    return application


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "👋 Bybit Funding Rate Scanner Bot'a hoş geldin!\n\n"
        "Bu bot, Bybit perpetual kontratlarda belirli kriterlere uyan coinleri "
        "her gün 09:00, 13:00 ve 17:00'de otomatik olarak tarar ve Telegram üzerinden bildirir.\n\n"
        "Komutlar:\n"
        "/status - Bot durumunu görüntüle\n"
        "/scan - Manuel tarama (sadece admin)\n"
        "/settings - Bildirim ayarları hakkında bilgi"
    )
    await update.message.reply_text(text)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: AppConfig = context.bot_data["config"]
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        now_tz = dt.datetime.now(ZoneInfo(config.timezone)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        now_tz = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = (
        "📡 *Bot Durumu*\n"
        f"Zaman (UTC): `{now_utc}`\n"
        f"Zaman ({config.timezone}): `{now_tz}`\n"
        f"Bybit Base URL: `{config.bybit.base_url}`\n"
        f"Kategori: `{config.bybit.category}`\n"
        f"Zaman dilimi: `{config.timezone}`\n"
        "\nOtomatik tarama 09:00, 13:00 ve 17:00 saatlerinde çalışacak."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "⚙️ *Bildirim Ayarları*\n\n"
        "- Bildirimler varsayılan olarak tek bir chat ID'ye gönderilir.\n"
        "- `.env` dosyasında `TELEGRAM_CHAT_ID` ve `ADMIN_USER_ID` ayarlarını güncelleyerek değiştirebilirsin.\n"
        "- Gelecekte çoklu kullanıcı ve favori coin listesi desteği eklenebilir."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


def _is_admin(update: Update, config: AppConfig) -> bool:
    user = update.effective_user
    if not user:
        return False
    # Prefer multi-admin list if available
    admin_ids = getattr(config.telegram, "admin_ids", None) or []
    if admin_ids:
        return user.id in admin_ids
    if config.telegram.admin_user_id is not None:
        return user.id == config.telegram.admin_user_id
    # If admin not configured, allow nobody
    return False


async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: AppConfig = context.bot_data["config"]
    if not _is_admin(update, config):
        await update.message.reply_text("❌ Bu komutu kullanma yetkin yok.")
        return

    from main import run_scan_once  # lazy import to avoid circular

    await update.message.reply_text("🔍 Manuel tarama başlatılıyor, lütfen bekleyin...")

    try:
        summary = await run_scan_once(context.application.bot_data, mode="full")
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Tarama sırasında hata oluştu: {exc}")
        return

    text = format_scan_notification(summary)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def scan_rise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: AppConfig = context.bot_data["config"]
    if not _is_admin(update, config):
        await update.message.reply_text("❌ Bu komutu kullanma yetkin yok.")
        return

    from main import run_scan_once  # lazy import to avoid circular

    await update.message.reply_text("🔍 Yükseliş taraması başlatılıyor, lütfen bekleyin...")

    try:
        summary = await run_scan_once(context.application.bot_data, mode="rise_only")
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Tarama sırasında hata oluştu: {exc}")
        return

    text = format_scan_notification(summary, funding_filter_applied=False)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def scan_fall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: AppConfig = context.bot_data["config"]
    if not _is_admin(update, config):
        await update.message.reply_text("❌ Bu komutu kullanma yetkin yok.")
        return

    from main import run_scan_once  # lazy import to avoid circular

    await update.message.reply_text("🔍 Düşüş taraması başlatılıyor, lütfen bekleyin...")

    try:
        summary = await run_scan_once(context.application.bot_data, mode="fall_only")
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Tarama sırasında hata oluştu: {exc}")
        return

    text = format_scan_notification(summary, funding_filter_applied=False)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


def format_scan_notification(summary: ScanSummary, funding_filter_applied: bool = True) -> str:
    ts_str = summary.timestamp.strftime("%H:%M")
    funding_line = "• Actual Funding (son): Negatif (<0%)\n" if funding_filter_applied else ""
    header = (
        "🚨 *ÇITA SAATİ ALARM* 🚨\n"
        f"⏰ Zaman: *{ts_str}*\n"
        f"📊 Toplam Taranan: *{summary.total_scanned}* Coin\n"
        f"✅ Uygun Coin: *{summary.matched_count}* Adet\n"
        "🎯 Kriterler:\n"
        "• Funding Interval: 1s veya 4s\n"
        f"{funding_line}"
        f"• Anlık Yükseliş: +{summary.matches[0]['price_change_pct']:.2f}% ve üzeri"
        if summary.matches
        else "• Anlık Yükseliş: +2% / +3%\n"
    )

    lines = [header, "📈 Uygun Coinler:"]

    if not summary.matches:
        lines.append("_Bu sefer kriterlere uyan coin bulunamadı._")
    else:
        for idx, m in enumerate(summary.matches, start=1):
            symbol = m["symbol"]
            price = m["last_price"]
            change = m["price_change_pct"]
            funding_val = m.get("funding_rate")
            funding_rate = (funding_val * 100) if funding_val is not None else None
            interval_min = m.get("funding_interval_min")
            interval_hours = (interval_min / 60) if interval_min else None
            interval_text = f"⏱️ Interval: {interval_hours:g} saat\n" if interval_hours else ""

            funding_text = f"📉 Funding: {funding_rate:.4f}%\n" if funding_rate is not None else ""
            ml_line = ""
            if "long_prob" in m and m["long_prob"] is not None:
                pct = round(m["long_prob"] * 100)
                ml_line = f"🤖 *Long ihtimali: %{pct}* (Short: %{100 - pct})\n"
            lines.append(
                f"{idx}️⃣ *{symbol}*\n"
                f"💰 Fiyat: `${price:,.4f}` (+{change:.2f}%)\n"
                f"{funding_text}"
                f"{ml_line}"
                f"{interval_text}"
            )

    lines.append("🔗 Bybit Funding Sayfası: `https://www.bybit.com/funding-rate`")
    return "\n".join(lines)


async def send_scan_notification(
    bot_data: dict,
    summary: ScanSummary,
) -> None:
    """
    Used by scheduler to send notifications without a user command.
    """
    config: AppConfig = bot_data["config"]
    application: Application = bot_data["application"]

    if config.telegram.chat_id is None:
        # No chat configured – nothing to send to
        return

    text = format_scan_notification(summary)
    await application.bot.send_message(
        chat_id=config.telegram.chat_id,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
    )

