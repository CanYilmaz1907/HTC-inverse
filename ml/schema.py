"""ML veri şeması — eğitim, canlı öğrenme ve inference aynı özellik sırasını kullanır."""

from __future__ import annotations

FEATURE_NAMES = [
    "funding_rate",
    "funding_rate_prev",
    "funding_change",
    "oi_5m_change",
    "oi_5m_level",
    "ls_buy_ratio",
    "ls_sell_ratio",
    "orderbook_imbalance",
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

META_COLUMNS = ["symbol", "ts", "target_15m_pct", "label"]
DATASET_COLUMNS = META_COLUMNS + FEATURE_NAMES
