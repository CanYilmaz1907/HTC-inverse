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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    base = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    # Calibrate probabilities so % values are more meaningful
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train_scaled, y_train)

    val_proba = clf.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_acc = float((val_pred == y_val.to_numpy()).mean())

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_cols, "val_acc": val_acc, "n_train": int(len(y_train)), "n_val": int(len(y_val))}, f, indent=2)

    print(f"Saved model to {MODEL_PATH}, scaler to {SCALER_PATH}, features: {feature_cols}")
    print(f"Train samples: {len(y)}, Long%: {y.mean()*100:.1f}")
    print(f"Val acc: {val_acc:.3f}")


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
