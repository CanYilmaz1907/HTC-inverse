"""
Canlı sembol için model_meta.json ile uyumlu özellik vektörü üretir.
Bazı uçlar (account-ratio, OI) borsa/sembol bazında başarısız olabilir; güvenli varsayılanlar kullanılır.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional

from bybit_client import BybitClient
from scanner import _get_last_5m_change_pct


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_pct(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    n = len(closes)
    if n < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, n):
        h, low, prev_c = highs[i], lows[i], closes[i - 1]
        tr = max(h - low, abs(h - prev_c), abs(low - prev_c))
        trs.append(tr)
    atr = sum(trs[-period:]) / period
    last = closes[-1]
    return (atr / last) * 100.0 if last > 0 else 0.0


async def _closed_candle_change_pct(
    client: BybitClient,
    symbol: str,
    sem: asyncio.Semaphore,
    interval: str,
    minutes: int,
) -> Optional[float]:
    async with sem:
        klines = await client.get_kline(symbol=symbol, interval=interval, limit=3)
    if not klines:
        return None
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    span_ms = minutes * 60 * 1000
    chosen = None
    for k in klines:
        if len(k) < 5:
            continue
        start_ms = _f(k[0])
        if start_ms is None:
            continue
        if int(start_ms) <= now_ms - span_ms:
            chosen = k
            break
    if chosen is None:
        chosen = klines[-1]
    o, c = _f(chosen[1]), _f(chosen[4])
    if o is None or c is None or o <= 0:
        return None
    return (c - o) / o * 100.0


async def build_ml_feature_row(
    client: BybitClient,
    symbol: str,
    tz: dt.tzinfo,
    sem: asyncio.Semaphore,
    feature_names: List[str],
) -> Optional[List[float]]:
    """model_meta.json feature_names sırasında vektör; kritik veri yoksa None."""
    async with sem:
        tickers = await client.get_tickers(symbols=[symbol])
    if not tickers:
        return None
    t = tickers[0]
    last = _f(t.get("lastPrice"))
    if last is None or last <= 0:
        return None
    high24 = _f(t.get("highPrice24h")) or last
    low24 = _f(t.get("lowPrice24h")) or last
    if high24 <= low24:
        pos_24 = 0.5
    else:
        pos_24 = (last - low24) / (high24 - low24)

    funding_rate = _f(t.get("fundingRate"))
    if funding_rate is None:
        funding_rate = 0.0

    funding_prev = funding_rate
    funding_change = 0.0
    try:
        async with sem:
            fh = await client.get_funding_history(symbol=symbol, limit=2)
        if len(fh) >= 2:
            fp = _f(fh[1].get("fundingRate"))
            if fp is not None:
                funding_prev = fp
                funding_change = funding_rate - funding_prev
    except Exception:
        pass

    oi_change = 0.0
    oi_level = 0.0
    try:
        async with sem:
            oi_list = await client.get_open_interest(symbol=symbol, interval_time="5min", limit=3)
        if len(oi_list) >= 2:
            cur = _f(oi_list[0].get("openInterest"))
            prev = _f(oi_list[1].get("openInterest"))
            if cur is not None:
                oi_level = cur
            if cur is not None and prev is not None and prev != 0:
                oi_change = (cur - prev) / prev
    except Exception:
        pass

    ls_buy = 0.5
    ls_sell = 0.5
    try:
        async with sem:
            ratios = await client.get_account_ratio(symbol=symbol, period="5min", limit=2)
        if ratios:
            r0 = ratios[0]
            b = _f(r0.get("buyRatio"))
            s = _f(r0.get("sellRatio"))
            if b is not None and s is not None:
                ls_buy, ls_sell = b, s
    except Exception:
        pass

    ob_imb = 0.0
    try:
        async with sem:
            ob = await client.get_orderbook(symbol=symbol, limit=50)
        bids = ob.get("b") or ob.get("bids") or []
        asks = ob.get("a") or ob.get("asks") or []
        bv = sum(_f(x[1]) or 0.0 for x in bids if len(x) >= 2)
        av = sum(_f(x[1]) or 0.0 for x in asks if len(x) >= 2)
        if bv + av > 0:
            ob_imb = (bv - av) / (bv + av)
    except Exception:
        pass

    change_5m = await _get_last_5m_change_pct(client, symbol, sem)
    if change_5m is None:
        return None

    change_15m = await _closed_candle_change_pct(client, symbol, sem, "15", 15)
    change_1h = await _closed_candle_change_pct(client, symbol, sem, "60", 60)
    change_4h = await _closed_candle_change_pct(client, symbol, sem, "240", 240)
    if change_15m is None:
        change_15m = change_5m
    if change_1h is None:
        change_1h = change_5m
    if change_4h is None:
        change_4h = change_5m

    async with sem:
        k5 = await client.get_kline(symbol=symbol, interval="5", limit=60)
    if not k5 or len(k5) < 5:
        return None

    series = list(reversed(k5))
    opens = [_f(x[1]) for x in series]
    highs = [_f(x[2]) for x in series]
    lows = [_f(x[3]) for x in series]
    closes = [_f(x[4]) for x in series]
    vols = [_f(x[5]) for x in series]
    if any(v is None for v in opens + highs + lows + closes):
        return None
    opens = [float(x) for x in opens]  # type: ignore
    highs = [float(x) for x in highs]
    lows = [float(x) for x in lows]
    closes_c = [float(x) for x in closes]

    vol_ratio = 0.0
    vv = [v for v in vols if v is not None]
    if len(vv) >= 13:
        vv = [float(x) for x in vv]  # type: ignore
        cur_v = vv[-1]
        mean_p = sum(vv[-13:-1]) / 12.0
        if mean_p > 0:
            vol_ratio = cur_v / mean_p - 1.0

    rsi_v = _rsi(closes_c, 14)
    atr_v = _atr_pct(highs, lows, closes_c, 14)

    consec = 0.0
    for i in range(len(closes_c) - 2, -1, -1):
        if closes_c[i] > opens[i]:
            consec += 1.0
        else:
            break

    prev_5m = 0.0
    if len(opens) >= 2 and opens[-2] > 0:
        prev_5m = (closes_c[-2] - opens[-2]) / opens[-2] * 100.0

    now_local = dt.datetime.now(tz=tz)
    hour = float(now_local.hour)
    dow = float(now_local.weekday())

    row: Dict[str, float] = {
        "funding_rate": funding_rate,
        "funding_rate_prev": funding_prev,
        "funding_change": funding_change,
        "oi_5m_change": oi_change,
        "oi_5m_level": oi_level,
        "ls_buy_ratio": ls_buy,
        "ls_sell_ratio": ls_sell,
        "orderbook_imbalance": ob_imb,
        "change_5m": change_5m,
        "change_15m": change_15m,
        "change_1h": change_1h,
        "change_4h": change_4h,
        "volume_ratio": vol_ratio,
        "position_in_24h_range": pos_24,
        "rsi_14_5m": rsi_v,
        "atr_14_pct": atr_v,
        "consecutive_green_5m": consec,
        "prev_5m_return": prev_5m,
        "hour": hour,
        "day_of_week": dow,
    }

    try:
        return [float(row[name]) for name in feature_names]
    except KeyError:
        return None
