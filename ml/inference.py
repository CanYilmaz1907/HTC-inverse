from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np


class MLTradingSignals:
    """Eğitilmiş RandomForest + StandardScaler ile yukarı sınıfı (label=1) olasılığı."""

    def __init__(self, ml_dir: Optional[Path] = None) -> None:
        self._root = ml_dir or Path(__file__).resolve().parent
        self.feature_names: List[str] = []
        self._model = None
        self._scaler = None
        self.meta: dict = {}
        self.reload()

    def reload(self) -> None:
        meta_path = self._root / "model_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"model_meta.json bulunamı: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_names = list(self.meta["feature_names"])
        self._model = joblib.load(self._root / "model.joblib")
        self._scaler = joblib.load(self._root / "scaler.joblib")

    def proba_up(self, features: List[float]) -> float:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Özellik sayısı uyuşmuyor: {len(features)} != {len(self.feature_names)}"
            )
        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            Xs = self._scaler.transform(X)
            proba = self._model.predict_proba(Xs)[0]
        # İkili sınıf: classes_ == [0, 1] → indeks 1 = yukarı (label=1)
        return float(proba[1])
