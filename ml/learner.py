"""Sürekli öğrenme: gözlem kaydı, etiketleme, periyodik yeniden eğitim."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ml.samples import SampleStore
from ml.schema import FEATURE_NAMES
from ml.train import train_and_save

if TYPE_CHECKING:
    from ml.inference import MLTradingSignals


class ContinuousLearner:
    def __init__(self, ml_dir: Optional[Path] = None) -> None:
        self.root = ml_dir or Path(__file__).resolve().parent
        self.store = SampleStore(self.root)
        self.enabled = os.getenv("ML_AUTO_LEARN", "1").lower() in ("1", "true", "yes")
        self.label_minutes = int(os.getenv("ML_LABEL_MINUTES", "15"))
        self.retrain_min_samples = int(os.getenv("ML_RETRAIN_MIN_SAMPLES", "20"))
        self.retrain_hours = float(os.getenv("ML_RETRAIN_HOURS", "12"))
        self.observe_max = int(os.getenv("ML_OBSERVE_MAX", "5"))
        self._samples_at_last_train = self.store.count_live_samples()
        self._last_train_ts = 0.0
        self._pending_labeled_total = 0

    def observe_candidate(
        self,
        *,
        symbol: str,
        entry_price: float,
        features: Dict[str, float],
        p_up: float,
        source: str,
        side_hint: str,
    ) -> None:
        if not self.enabled:
            return
        ts_ms = int(time.time() * 1000)
        self.store.queue_observation(
            symbol=symbol,
            ts_ms=ts_ms,
            entry_price=entry_price,
            features=features,
            p_up=p_up,
            source=source,
            side_hint=side_hint,
        )

    async def maintenance(
        self,
        client: Any,
        ml_signals: Optional["MLTradingSignals"] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        labeled = await self.store.process_pending(client, label_minutes=self.label_minutes)
        if labeled:
            self._pending_labeled_total += labeled
            print(f"📚 ML: {labeled} yeni etiket eklendi (live_samples.csv)")

        live_count = self.store.count_live_samples()
        new_since = live_count - self._samples_at_last_train
        hours_ok = (time.time() - self._last_train_ts) >= self.retrain_hours * 3600.0
        if new_since < self.retrain_min_samples and not (hours_ok and new_since > 0):
            return None

        try:
            meta = train_and_save(self.root)
            self._samples_at_last_train = live_count
            self._last_train_ts = time.time()
            print(
                f"🧠 ML yeniden eğitildi | val_acc={meta['val_acc']:.3f} | "
                f"toplam örnek={meta['n_total']} | yeni etiket={new_since}"
            )
            if ml_signals is not None:
                ml_signals.reload()
                print("♻️ ML modeli canlıya yüklendi (hot-reload).")
            return meta
        except Exception as e:
            print(f"⚠️ ML yeniden eğitim atlandı: {e}")
            return None

    def pick_observations(
        self,
        scored: List[tuple],
    ) -> List[tuple]:
        """En güvenilir adaylardan öğrenme için üst N seç."""
        if not scored:
            return []
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        return ranked[: max(1, self.observe_max)]
