from __future__ import annotations

import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from bybit_client import BybitClient
from config import AppConfig
from telegram_handler import send_scan_notification


def _get_timezone(config: AppConfig) -> dt.tzinfo:
    tz_name = (config.timezone or "UTC").strip()
    if tz_name.upper() == "UTC":
        return dt.timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


async def setup_scheduler(app_bot_data: dict) -> AsyncIOScheduler:
    """
    Schedule cron jobs for 09:00 / 13:00 / 17:00 scans.
    """
    config: AppConfig = app_bot_data["config"]
    client: BybitClient = app_bot_data["bybit_client"]

    # Store app loop so scheduler jobs (running in another thread) can schedule coroutines on it
    app_bot_data["event_loop"] = asyncio.get_running_loop()

    tz = _get_timezone(config)

    scheduler = AsyncIOScheduler(timezone=tz)

    signal_hours = (9, 13, 17)

    def _hhmm(h: int, m: int) -> str:
        return f"{h:02d}:{m:02d}"

    def _make_scan_job(signal_label: str):
        async def _job() -> None:
            from main import run_scan_once  # avoid circular

            try:
                summary = await run_scan_once(app_bot_data, mode="full")
                await send_scan_notification(app_bot_data, summary)
            except Exception as exc:  # noqa: BLE001
                await _notify_admin_error(app_bot_data, f"{signal_label} tarama hatası: {exc}")

        return _job

    def wrap(coro_func):
        def runner():
            loop = app_bot_data["event_loop"]
            asyncio.run_coroutine_threadsafe(coro_func(), loop)

        return runner

    for h in signal_hours:
        signal_label = _hhmm(h, 0)

        scheduler.add_job(
            wrap(_make_scan_job(signal_label)),
            CronTrigger(hour=h, minute=0, second=0, timezone=tz),
            id=f"scan_and_notify_{h:02d}00",
            replace_existing=True,
            misfire_grace_time=120,
            coalesce=True,
            max_instances=1,
        )

    # Optional near-real-time scan (every N minutes) with high-confidence filter
    if getattr(config.criteria, "realtime_scan_enabled", False):
        every = getattr(config.criteria, "realtime_scan_every_minutes", 5) or 5
        min_conf = getattr(config.criteria, "realtime_min_confidence", 0.7) or 0.7

        def _make_realtime_job():
            async def _job() -> None:
                from main import run_scan_once  # avoid circular

                try:
                    summary = await run_scan_once(app_bot_data, mode="full")
                    # Send only if any match has high confidence
                    keep = []
                    for m in summary.matches:
                        p = m.get("long_prob")
                        if p is None:
                            continue
                        if p >= min_conf or p <= (1.0 - min_conf):
                            keep.append(m)
                    summary.matches = keep
                    summary.matched_count = len(keep)
                    if keep:
                        await send_scan_notification(app_bot_data, summary)
                except Exception as exc:  # noqa: BLE001
                    await _notify_admin_error(app_bot_data, f"realtime tarama hatası: {exc}")

            return _job

        scheduler.add_job(
            wrap(_make_realtime_job()),
            CronTrigger(minute=f"*/{every}", second=5, timezone=tz),
            id="scan_and_notify_realtime",
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
            max_instances=1,
        )

    scheduler.start()
    return scheduler


async def _notify_admin_error(bot_data: dict, message: str) -> None:
    from telegram_handler import Application

    config: AppConfig = bot_data["config"]
    application: Application = bot_data["application"]

    admin_ids = getattr(config.telegram, "admin_ids", None) or []
    admin_id = admin_ids[0] if admin_ids else config.telegram.admin_user_id
    if admin_id is None:
        return

    await application.bot.send_message(chat_id=admin_id, text=f"⚠️ Bot hatası: {message}")

