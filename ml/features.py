"""
Feature extraction for ML: funding, multi-timeframe returns, volume, RSI, ATR, time.
All values numeric; missing filled with 0 or safe default for model.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from bybit_client import BybitClient


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_return(open_p: Optional[float], close_p: Optional[float]) -> Optional[float]:
    if open_p is None or close_p is None or open_p <= 0:
        return None
    return (close_p - open_p) / open_p * 100.0


def _rsi_from_closes(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    # deltas: change from previous close to current (closes[i] - closes[i+1] = change into i+1)
    deltas = [closes[i] - closes[i + 1] for i in range(len(closes) - 1)]
    use = deltas[-period:]
    gains = [d for d in use if d > 0]
    losses = [-d for d in use if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _atr_pct(highs: List[float], lows: List[float], closes: List[float], period: int, last_close: float) -> float:
    if len(closes) < period + 1 or last_close <= 0:
        return 0.0
    tr_list = []
    for i in range(len(closes) - 1):
        h = highs[i] if i < len(highs) else closes[i]
        l = lows[i] if i < len(lows) else closes[i]
        prev_close = closes[i + 1]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        tr_list.append(tr)
    if len(tr_list) < period:
        return 0.0
    atr = sum(tr_list[-period:]) / period
    return (atr / last_close) * 100.0


# Feature keys we use in model (order for DataFrame)
FEATURE_NAMES = [
    "funding_rate",
    "funding_rate_prev",
    "funding_change",
    "change_5m",
    "change_15m",
    "change_1h",
    "change_4h",
    "volume_ratio",
    "position_in_24h_range",
    "rsi_14_5m",
    "atr_14_pct",
    "consecutive_green_5m",
    "prev_5m_return",
    "hour",
    "day_of_week",
]


async def extract_features_for_match(
    client: BybitClient,
    symbol: str,
    current_price: float,
    change_5m: float,
    funding_rate: float,
    tz: dt.tzinfo,
    *,
    now: Optional[dt.datetime] = None,
) -> Dict[str, float]:
    """
    Extract all ML features for a scan match at 'now' (or current time).
    Uses klines 5/15/60/240, ticker for 24h range/volume, funding history for prev.
    """
    if now is None:
        now = dt.datetime.now(tz)
    now_utc = now.astimezone(dt.timezone.utc)
    now_ms = int(now_utc.timestamp() * 1000)
    five_min_ms = 5 * 60 * 1000
    # Last closed 5m candle ends at: floor(now/5m)*5m - 5m
    end_5m = (now_ms // five_min_ms) * five_min_ms - five_min_ms
    start_5m = end_5m - 20 * five_min_ms  # 20 candles for RSI/ATR/prev

    out: Dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}
    out["funding_rate"] = funding_rate
    out["change_5m"] = change_5m

    # Funding prev
    try:
        fund_hist = await client.get_funding_history(symbol=symbol, limit=3)
        if len(fund_hist) >= 2:
            prev = _parse_float(fund_hist[1].get("fundingRate"))
            if prev is not None:
                out["funding_rate_prev"] = prev
                out["funding_change"] = funding_rate - prev
    except Exception:
        pass

    # 5m klines (for RSI, ATR, consecutive green, prev_5m)
    try:
        klines_5 = await client.get_kline(symbol=symbol, interval="5", limit=25, start_time=int(start_5m), end_time=int(end_5m + 1))
        if klines_5:
            # Bybit newest first; we want chronological for RSI so reverse
            klines_5 = list(reversed(klines_5))
            opens = [_parse_float(k[1]) for k in klines_5 if len(k) >= 5]
            closes = [_parse_float(k[4]) for k in klines_5 if len(k) >= 5]
            highs = [_parse_float(k[2]) for k in klines_5 if len(k) >= 3]
            lows = [_parse_float(k[3]) for k in klines_5 if len(k) >= 4]
            volumes = [_parse_float(k[5]) for k in klines_5 if len(k) >= 6]
            closes_f = [c for c in closes if c is not None]
            if len(closes_f) >= 2:
                out["prev_5m_return"] = _pct_return(opens[-2], closes[-2]) or 0.0
            if len(closes_f) >= 15:
                out["rsi_14_5m"] = _rsi_from_closes(closes_f[-15:], 14)
            if len(closes_f) >= 15 and highs and lows:
                out["atr_14_pct"] = _atr_pct(highs[-15:], lows[-15:], closes_f[-15:], 14, closes_f[-1])
            # Consecutive green (from end backwards)
            n_green = 0
            for i in range(len(opens) - 1, -1, -1):
                if opens[i] and closes[i] and closes[i] >= opens[i]:
                    n_green += 1
                else:
                    break
            out["consecutive_green_5m"] = float(n_green)
            if volumes and volumes[-1] is not None:
                out["volume_5m_raw"] = volumes[-1]  # for volume_ratio below
    except Exception:
        pass

    # 15m, 1h, 4h returns (last closed candle each)
    for interval, key in [("15", "change_15m"), ("60", "change_1h"), ("240", "change_4h")]:
        try:
            interval_min = int(interval)
            interval_ms = interval_min * 60 * 1000
            end_i = (now_ms // interval_ms) * interval_ms - interval_ms
            start_i = end_i - 2 * interval_ms
            klines = await client.get_kline(symbol=symbol, interval=interval, limit=2, start_time=int(start_i), end_time=int(end_i + 1))
            if klines and len(klines) >= 1:
                k = klines[0]
                if len(k) >= 5:
                    o, c = _parse_float(k[1]), _parse_float(k[4])
                    out[key] = _pct_return(o, c) or 0.0
        except Exception:
            pass

    # Ticker: 24h high, low, volume
    try:
        tickers = await client.get_tickers(symbols=[symbol])
        if tickers and len(tickers) >= 1:
            t = tickers[0]
            high_24h = _parse_float(t.get("highPrice24h"))
            low_24h = _parse_float(t.get("lowPrice24h"))
            vol_24h = _parse_float(t.get("volume24h"))
            if high_24h is not None and low_24h is not None and high_24h > low_24h:
                out["position_in_24h_range"] = (current_price - low_24h) / (high_24h - low_24h)
                out["position_in_24h_range"] = max(0.0, min(1.0, out["position_in_24h_range"]))
            if vol_24h is not None and vol_24h > 0 and "volume_5m_raw" in out:
                # 288 = 5m candles in 24h
                out["volume_ratio"] = out["volume_5m_raw"] / (vol_24h / 288)
    except Exception:
        pass
    if "volume_5m_raw" in out:
        del out["volume_5m_raw"]

    out["hour"] = float(now_utc.hour)
    out["day_of_week"] = float(now_utc.weekday())

    return {k: out.get(k, 0.0) for k in FEATURE_NAMES}


def feature_vector_for_model(features: Dict[str, float]) -> List[float]:
    """Return feature list in FEATURE_NAMES order for model."""
    return [features.get(k, 0.0) for k in FEATURE_NAMES]
