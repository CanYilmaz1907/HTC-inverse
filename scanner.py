import datetime as dt
from dataclasses import dataclass
import asyncio
from typing import Any, Dict, List, Optional

from bybit_client import BybitClient
from config import ScannerCriteria


@dataclass
class ScanSummary:
    total_scanned: int
    matched_count: int
    timestamp: dt.datetime
    matches: List[Dict[str, Any]]


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


async def _get_latest_actual_funding_rate(
    client: BybitClient,
    symbol: str,
    sem: asyncio.Semaphore,
) -> Optional[float]:
    """
    Returns the latest settled funding rate (Actual Funding) from funding history.
    """
    async with sem:
        items = await client.get_funding_history(symbol=symbol, limit=1)
    if not items:
        return None
    return _parse_float(items[0].get("fundingRate"))


async def _get_last_5m_change_pct(
    client: BybitClient,
    symbol: str,
    sem: asyncio.Semaphore,
) -> Optional[float]:
    """
    Computes percent change of the latest *closed* 5m candle: (close-open)/open*100.

    At exact boundaries (e.g. 17:00:00), the most recent kline can be the newly opened
    candle (17:00–17:05) with near-zero change. We therefore fetch 2 candles and choose
    the latest candle that is safely closed.
    """
    async with sem:
        klines = await client.get_kline(symbol=symbol, interval="5", limit=2)
    if not klines:
        return None

    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    five_min_ms = 5 * 60 * 1000

    # Bybit returns newest first. Pick the newest candle whose startTime <= now-5m.
    chosen = None
    for k in klines:
        if len(k) < 1:
            continue
        start_ms = _parse_float(k[0])
        if start_ms is None:
            continue
        if int(start_ms) <= now_ms - five_min_ms:
            chosen = k
            break

    if chosen is None:
        # Fallback: use the older candle if present
        chosen = klines[-1]

    k = chosen
    if len(k) < 5:
        return None
    open_p = _parse_float(k[1])
    close_p = _parse_float(k[4])
    if open_p is None or close_p is None or open_p <= 0:
        return None
    return (close_p - open_p) / open_p * 100.0


async def run_scan(
    client: BybitClient,
    criteria: ScannerCriteria,
    tz: dt.tzinfo,
    *,
    require_actual_funding_negative: bool = True,
    direction: str = "up",  # "up" | "down" | "both" (mutlak 5m hareket >= eşik)
) -> ScanSummary:
    """
    Instant scan at the current moment.
    Uses Bybit tickers 24h change (price24hPcnt) as "yükseliş".
    """
    now = dt.datetime.now(tz=tz)
    instruments = await client.get_instruments_info()
    funding_intervals: Dict[str, int] = {}
    perpetual_symbols: set[str] = set()

    for inst in instruments:
        if inst.get("status") != "Trading":
            continue
        contract_type = inst.get("contractType") or ""
        if str(contract_type).lower() not in {"perpetual", "linearperpetual"}:
            continue

        symbol = inst.get("symbol")
        if not symbol:
            continue

        perpetual_symbols.add(symbol)
        # fundingInterval bilgisini artık filtre için kullanmıyoruz,
        # sadece enformasyon amaçlı saklıyoruz (varsa).
        interval_min_raw = inst.get("fundingInterval")
        try:
            interval_min = int(interval_min_raw)
        except (TypeError, ValueError):
            interval_min = None
        if interval_min is not None:
            funding_intervals[symbol] = interval_min

    tickers = await client.get_tickers()
    total_scanned = len([t for t in tickers if t.get("symbol") in perpetual_symbols])
    matches: List[Dict[str, Any]] = []

    # First pass: cheap filters using tickers + instruments metadata
    candidates: List[Dict[str, Any]] = []
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol or symbol not in perpetual_symbols:
            continue

        last_price = _parse_float(t.get("lastPrice"))
        if last_price is None or last_price <= 0:
            continue

        interval_min = funding_intervals.get(symbol)
        candidates.append(
            {
                "symbol": symbol,
                "last_price": last_price,
                "funding_interval_min": interval_min,
            }
        )

    # Second pass: fetch 5m change + (optional) actual funding rate for candidates
    sem = asyncio.Semaphore(15)  # keep under Bybit rate limit comfortably
    change_tasks = [_get_last_5m_change_pct(client, c["symbol"], sem) for c in candidates]
    changes = await asyncio.gather(*change_tasks, return_exceptions=True)

    funding_rates: List[object] = []
    if require_actual_funding_negative:
        funding_tasks = [_get_latest_actual_funding_rate(client, c["symbol"], sem) for c in candidates]
        funding_rates = await asyncio.gather(*funding_tasks, return_exceptions=True)
    else:
        funding_rates = [None for _ in candidates]

    for c, chg, fr in zip(candidates, changes, funding_rates):
        if isinstance(chg, Exception):
            continue
        if chg is None:
            continue
        if direction == "up":
            if chg < criteria.min_price_change_percent:
                continue
        elif direction == "down":
            if chg > -criteria.min_price_change_percent:
                continue
        elif direction == "both":
            if abs(chg) < criteria.min_price_change_percent:
                continue
        else:
            continue
        if require_actual_funding_negative:
            if isinstance(fr, Exception):
                continue
            if fr is None or fr >= 0:
                continue
        c["price_change_pct"] = chg
        if require_actual_funding_negative:
            c["funding_rate"] = fr
        matches.append(c)

    # Sıralama: iki yönlü taramada mutlak hareket; tek yönde klasik sıra
    if direction == "both":
        matches.sort(key=lambda m: abs(m["price_change_pct"]), reverse=True)
    else:
        matches.sort(key=lambda m: m["price_change_pct"], reverse=True)

    return ScanSummary(
        total_scanned=total_scanned,
        matched_count=len(matches),
        timestamp=now,
        matches=matches,
    )

