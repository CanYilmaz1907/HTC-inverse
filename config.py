import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: Optional[int]
    admin_user_id: Optional[int]
    admin_ids: list[int]


@dataclass
class BybitConfig:
    base_url: str
    api_key: Optional[str]
    api_secret: Optional[str]
    category: str = "linear"


@dataclass
class ScannerCriteria:
    min_price_change_percent: float = 2.0
    allowed_funding_intervals_min: tuple[int, int] = (60, 240)  # 1h or 4h
    realtime_scan_enabled: bool = False
    realtime_scan_every_minutes: int = 5
    realtime_min_confidence: float = 0.7  # 0.70 => only alert if Long>=70% or Short>=70%


@dataclass
class AppConfig:
    telegram: TelegramConfig
    bybit: BybitConfig
    db_path: str
    criteria: ScannerCriteria
    timezone: str = "UTC"


def _get_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_admin_ids(single_id: Optional[int]) -> list[int]:
    raw = os.getenv("ADMIN_USER_IDS", "")
    ids: list[int] = []
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
    if single_id is not None and single_id not in ids:
        ids.append(single_id)
    return ids


def load_config() -> AppConfig:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment or .env file.")

    admin_user_id = _get_int_env("ADMIN_USER_ID")
    admin_ids = _parse_admin_ids(admin_user_id)

    telegram = TelegramConfig(
        bot_token=bot_token,
        chat_id=_get_int_env("TELEGRAM_CHAT_ID"),
        admin_user_id=admin_user_id,
        admin_ids=admin_ids,
    )

    bybit_base_url = os.getenv("BYBIT_BASE_URL", "https://api-testnet.bybit.com")
    bybit = BybitConfig(
        base_url=bybit_base_url.rstrip("/"),
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
    )

    db_path = os.getenv("DB_PATH", "bybit_funding_bot.db")

    min_change = float(os.getenv("MIN_PRICE_CHANGE_PERCENT", "2.0"))
    realtime_enabled = os.getenv("REALTIME_SCAN_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    realtime_every = int(os.getenv("REALTIME_SCAN_EVERY_MINUTES", "5"))
    realtime_conf = float(os.getenv("REALTIME_MIN_CONFIDENCE", "0.7"))
    criteria = ScannerCriteria(
        min_price_change_percent=min_change,
        realtime_scan_enabled=realtime_enabled,
        realtime_scan_every_minutes=max(1, realtime_every),
        realtime_min_confidence=max(0.5, min(0.99, realtime_conf)),
    )

    timezone = os.getenv("APP_TIMEZONE", "UTC")

    return AppConfig(
        telegram=telegram,
        bybit=bybit,
        db_path=db_path,
        criteria=criteria,
        timezone=timezone,
    )

