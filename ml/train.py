"""
Train classifier (Long=1, Short=0) from dataset.csv.
Saves model.joblib and scaler.joblib under ml/; used by predict.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
META_PATH = MODEL_DIR / "model_meta.json"


def train_and_save(dataset_path: Path) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns or df.empty:
        raise ValueError("dataset must contain 'label' and be non-empty")

    # Use only columns that exist and are numeric features
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in ["funding_rate", "change_5m", "funding_rate_prev", "funding_change"] if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in dataset")

    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_cols}, f, indent=2)

    print(f"Saved model to {MODEL_PATH}, scaler to {SCALER_PATH}, features: {feature_cols}")
    print(f"Train samples: {len(y)}, Long%: {y.mean()*100:.1f}")


def load_model_and_scaler():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None, []
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = {}
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    feature_names = meta.get("feature_names", FEATURE_NAMES)
    return clf, scaler, feature_names


if __name__ == "__main__":
    import sys
    path = MODEL_DIR / "dataset.csv"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    if not path.exists():
        print("Usage: python -m ml.train [dataset.csv path]")
        sys.exit(1)
    train_and_save(path)
