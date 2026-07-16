"""Canlı gözlemleri saklar, 15 dk sonra etiketleyip eğitim verisine ekler."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.schema import DATASET_COLUMNS, FEATURE_NAMES


class SampleStore:
    def __init__(self, ml_dir: Optional[Path] = None) -> None:
        self.root = ml_dir or Path(__file__).resolve().parent
        self.pending_path = self.root / "pending_observations.jsonl"
        self.live_path = self.root / "live_samples.csv"

    def _ensure_live_header(self) -> None:
        if self.live_path.is_file():
            return
        with self.live_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(DATASET_COLUMNS)

    def queue_observation(
        self,
        *,
        symbol: str,
        ts_ms: int,
        entry_price: float,
        features: Dict[str, float],
        p_up: float,
        source: str,
        side_hint: str = "",
    ) -> None:
        row = {
            "symbol": symbol,
            "ts_ms": ts_ms,
            "entry_price": entry_price,
            "features": {k: float(features[k]) for k in FEATURE_NAMES if k in features},
            "p_up": p_up,
            "source": source,
            "side_hint": side_hint,
            "labeled": False,
        }
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_pending(self) -> List[Dict[str, Any]]:
        if not self.pending_path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        with self.pending_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _rewrite_pending(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            if self.pending_path.is_file():
                self.pending_path.unlink()
            return
        with self.pending_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def process_pending(
        self,
        client: Any,
        *,
        label_minutes: int = 15,
    ) -> int:
        """Vadesi gelen gözlemleri etiketleyip live_samples.csv'ye yazar."""
        pending = self._load_pending()
        if not pending:
            return 0

        now_ms = int(time.time() * 1000)
        wait_ms = label_minutes * 60 * 1000
        still_pending: List[Dict[str, Any]] = []
        labeled = 0
        self._ensure_live_header()

        for obs in pending:
            ts_ms = int(obs.get("ts_ms", 0))
            if now_ms - ts_ms < wait_ms:
                still_pending.append(obs)
                continue
            symbol = str(obs.get("symbol", ""))
            entry_price = float(obs.get("entry_price", 0.0))
            feats = obs.get("features") or {}
            if not symbol or entry_price <= 0 or len(feats) < len(FEATURE_NAMES):
                continue
            try:
                tickers = await client.get_tickers(symbols=[symbol])
                lst = tickers if isinstance(tickers, list) else []
                if not lst:
                    still_pending.append(obs)
                    continue
                mark = float(lst[0].get("lastPrice") or 0.0)
                if mark <= 0:
                    still_pending.append(obs)
                    continue
            except Exception:
                still_pending.append(obs)
                continue

            target_pct = (mark - entry_price) / entry_price * 100.0
            label = 1 if target_pct > 0 else 0
            out = {
                "symbol": symbol,
                "ts": ts_ms,
                "target_15m_pct": target_pct,
                "label": label,
            }
            for name in FEATURE_NAMES:
                out[name] = float(feats.get(name, 0.0))

            with self.live_path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([out[col] for col in DATASET_COLUMNS])
            labeled += 1

        self._rewrite_pending(still_pending)
        return labeled

    def count_live_samples(self) -> int:
        if not self.live_path.is_file():
            return 0
        with self.live_path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
