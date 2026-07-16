"""dataset.csv + live_samples.csv ile modeli yeniden eğitir."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.schema import DATASET_COLUMNS, FEATURE_NAMES


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def load_training_frames(ml_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    root = ml_dir or Path(__file__).resolve().parent
    merged: Dict[str, Dict[str, Any]] = {}
    for fname in ("dataset.csv", "live_samples.csv"):
        for row in _read_rows(root / fname):
            key = f"{row.get('symbol')}|{row.get('ts')}"
            merged[key] = row

    rows = list(merged.values())
    if not rows:
        raise ValueError("Eğitim verisi yok (dataset.csv + live_samples.csv boş).")

    X: List[List[float]] = []
    y: List[int] = []
    for row in rows:
        try:
            label = int(float(row["label"]))
            feats = [float(row[name]) for name in FEATURE_NAMES]
        except (KeyError, TypeError, ValueError):
            continue
        X.append(feats)
        y.append(label)

    if len(X) < 40:
        raise ValueError(f"Eğitim için yeterli örnek yok ({len(X)} < 40).")

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int32), len(rows)


def train_and_save(ml_dir: Optional[Path] = None) -> Dict[str, Any]:
    root = ml_dir or Path(__file__).resolve().parent
    X, y, n_total = load_training_frames(root)

    stratify = y if len(set(y.tolist())) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    val_pred = model.predict(X_val_s)
    val_acc = float(accuracy_score(y_val, val_pred))

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    meta = {
        "feature_names": FEATURE_NAMES,
        "val_acc": val_acc,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_total": n_total,
        "class_counts": class_counts,
        "trained_at": int(time.time()),
        "model": "RandomForestClassifier",
    }

    joblib.dump(model, root / "model.joblib")
    joblib.dump(scaler, root / "scaler.joblib")
    (root / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    result = train_and_save()
    print(
        f"✅ Model eğitildi | val_acc={result['val_acc']:.3f} | "
        f"train={result['n_train']} val={result['n_val']} total={result['n_total']}"
    )
