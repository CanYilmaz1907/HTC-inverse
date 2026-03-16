"""
Rule-based Long/Short decision module.

Bu modül, her coin için aşağıdaki 7 kurala göre karar verir:
1) Long/Short oranı (lsr = buyRatio/sellRatio)
   - lsr < 0.4  -> LONG sinyali (+1)
   - lsr > 1.5  -> SHORT sinyali (-1)
2) Open Interest değişimi (oi_5m_change)
   - oi_change > 0.15 ve funding < 0  -> LONG (+1)
   - oi_change > 0.15 ve funding > 0  -> SHORT (-1)
3) RSI (rsi_14_5m)
   - rsi < 30  -> LONG (+1)
   - rsi > 70  -> SHORT (-1)
4) Funding Rate (funding_rate)
   - funding < -0.02 -> LONG (+1)
   - funding > 0.02  -> SHORT (-1)
5) Alış/Satış oranı (tekrar lsr kullanıyoruz)
   - lsr > 1.2 -> LONG (+1)
   - lsr < 0.8 -> SHORT (-1)
6) Fiyatın MA50'ye göre konumu
   - Fiyat MA50'den %5 aşağıda  -> LONG (+1)
   - Fiyat MA50'den %5 yukarıda -> SHORT (-1)
7) Total Score = önceki 6 kriterin (+1 / -1 / 0) toplamı.

En az 4 LONG kriteri varsa:
    decision = LONG, confidence >= 85
En az 4 SHORT kriteri varsa:
    decision = SHORT, confidence >= 85
Diğer durumlarda:
    - score >= 3  -> LONG (confidence ~70–85)
    - score <= -3 -> SHORT (confidence ~70–85)
    - aksi halde NÖTR (confidence düşük)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bybit_client import BybitClient
from config import AppConfig
from ml.features import extract_features_for_match, _parse_float


@dataclass
class LongShortDecision:
    symbol: str
    decision: str  # "LONG" | "SHORT" | "NÖTR"
    confidence: float  # 0-100
    reason: str
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "features": self.features,
        }


class RuleLongShortPredictor:
    """
    Rule-based predictor that uses ml.features + ek kline verisiyle
    belirtilen 7 kurala göre LONG/SHORT/NÖTR kararı üretir.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def _compute_ma50(
        self,
        client: BybitClient,
        symbol: str,
        tz: dt.tzinfo,
    ) -> Optional[float]:
        """
        Basit MA50 (50 adet 5m kapanış ortalaması).
        """
        try:
            now_utc = dt.datetime.now(dt.timezone.utc)
            now_ms = int(now_utc.timestamp() * 1000)
            five_min_ms = 5 * 60 * 1000
            end_5m = (now_ms // five_min_ms) * five_min_ms - five_min_ms
            start_5m = end_5m - 50 * five_min_ms
            klines = await client.get_kline(
                symbol=symbol,
                interval="5",
                limit=50,
                start_time=int(start_5m),
                end_time=int(end_5m + 1),
            )
        except Exception:
            return None
        if not klines:
            return None
        closes: List[float] = []
        for k in klines:
            if len(k) < 5:
                continue
            c = _parse_float(k[4])
            if c is not None:
                closes.append(c)
        if not closes:
            return None
        return sum(closes) / len(closes)

    async def evaluate_symbol(
        self,
        client: BybitClient,
        symbol: str,
        last_price: float,
        change_5m: float,
        funding_rate: float,
        tz: dt.tzinfo,
    ) -> Optional[LongShortDecision]:
        """
        Tek bir sembol için tüm özellikleri hesaplar ve karar döner.
        """
        try:
            feats = await extract_features_for_match(
                client,
                symbol,
                current_price=float(last_price),
                change_5m=float(change_5m),
                funding_rate=float(funding_rate),
                tz=tz,
            )
        except Exception:
            return None

        # Gerekli ham değerler
        funding = float(feats.get("funding_rate", 0.0))
        rsi = float(feats.get("rsi_14_5m", 50.0))
        oi_change = float(feats.get("oi_5m_change", 0.0))
        buy_ratio = float(feats.get("ls_buy_ratio", 0.0))
        sell_ratio = float(feats.get("ls_sell_ratio", 0.0))
        lsr = buy_ratio / sell_ratio if sell_ratio not in (0.0, 0) else 0.0

        # MA50
        ma50 = await self._compute_ma50(client, symbol, tz)
        ma_diff_pct = 0.0
        if ma50 and ma50 > 0:
            ma_diff_pct = (last_price - ma50) / ma50 * 100.0

        long_votes = 0
        short_votes = 0

        # 1) Long/Short oranı
        if lsr < 0.4:
            long_votes += 1
        elif lsr > 1.5:
            short_votes += 1

        # 2) Open Interest değişimi (trend devam)
        if oi_change > 0.15:
            if funding < 0:
                long_votes += 1
            elif funding > 0:
                short_votes += 1

        # 3) RSI
        if rsi < 30:
            long_votes += 1
        elif rsi > 70:
            short_votes += 1

        # 4) Funding rate
        if funding < -0.02:
            long_votes += 1
        elif funding > 0.02:
            short_votes += 1

        # 5) Alış/Satış oranı (lsr)
        if lsr > 1.2:
            long_votes += 1
        elif lsr < 0.8:
            short_votes += 1

        # 6) MA50 farkı
        if ma50 and ma50 > 0:
            if ma_diff_pct <= -5.0:
                long_votes += 1
            elif ma_diff_pct >= 5.0:
                short_votes += 1

        score = long_votes - short_votes

        decision = "NÖTR"
        confidence = 50.0
        reason = f"LONG={long_votes}, SHORT={short_votes}, score={score}"

        # 2. kural: 4+ aynı yönde ise doğrudan karar
        if long_votes >= 4 and long_votes > short_votes:
            decision = "LONG"
            confidence = min(100.0, 85.0 + (long_votes - 4) * 3.0)
            reason = f"{long_votes}/7 LONG kriteri sağlandı"
        elif short_votes >= 4 and short_votes > long_votes:
            decision = "SHORT"
            confidence = min(100.0, 85.0 + (short_votes - 4) * 3.0)
            reason = f"{short_votes}/7 SHORT kriteri sağlandı"
        else:
            # Total Score kuralı
            if score >= 3:
                decision = "LONG"
                confidence = min(90.0, 70.0 + (score - 3) * 5.0)
                reason = f"score={score} (LONG ağırlıklı)"
            elif score <= -3:
                decision = "SHORT"
                confidence = min(90.0, 70.0 + (-score - 3) * 5.0)
                reason = f"score={score} (SHORT ağırlıklı)"
            else:
                decision = "NÖTR"
                confidence = 50.0
                reason = f"score={score}, sinyal zayıf"

        features_out: Dict[str, Any] = {
            "lsr": round(lsr, 3),
            "rsi": round(rsi, 2),
            "funding": round(funding, 5),
            "oi_change": round(oi_change, 4),
            "ma50_diff_pct": round(ma_diff_pct, 2),
        }

        return LongShortDecision(
            symbol=symbol,
            decision=decision,
            confidence=confidence,
            reason=reason,
            features=features_out,
        )

