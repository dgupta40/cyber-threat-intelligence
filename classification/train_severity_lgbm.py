"""
train_severity_lgbm.py

Severity classifier - SBERT+LightGBM (balanced, oversampled)

Outputs
  • models/severity_lgbm_<ts>.pkl
  • metrics/lgbm_report_<ts>.json
  • data/processed/master_plus_pred_<ts>.parquet
"""

from __future__ import annotations
import glob, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── paths ───────────────────────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
METRICS_DIR= Path("metrics")
MODEL_DIR.mkdir(exist_ok=True); METRICS_DIR.mkdir(exist_ok=True)


def _train() -> None:

# ── 1. load latest parquet & SBERT matrix ───────────────────────────────
    parquet = max(glob.glob("data/processed/master_*.parquet"))
    df      = pd.read_parquet(parquet)
    embeddings = np.load(df["sbert_path"].iloc[0]).astype("float32")

    # keep only labelled rows
    mask = df["severity_bin"].isin(["critical", "high", "medium", "low"])
    df, embeddings = df[mask], embeddings[mask]

    le = LabelEncoder()
    y  = le.fit_transform(df["severity_bin"])

    # ── 2. oversample minority classes ──────────────────────────────────────
    ros = RandomOverSampler(random_state=42)
    X_bal, y_bal = ros.fit_resample(embeddings, y)

    # ── 3. train/test split ─────────────────────────────────────────────────
    Xtr, Xte, ytr, yte = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    # ── 4. fit LightGBM (balanced) ──────────────────────────────────────────
    clf = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            num_leaves=63,
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
    )
    clf.fit(Xtr, ytr)

    # ── 5. metrics ──────────────────────────────────────────────────────────
    report = classification_report(
                yte, clf.predict(Xte),
                target_names=le.classes_, output_dict=True)

    metrics_path = METRICS_DIR / f"lgbm_report_{datetime.utcnow():%Y%m%d_%H%M}.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    print("saved metrics ->", metrics_path)

    # ── 6. save model ───────────────────────────────────────────────────────
    model_path = MODEL_DIR / f"severity_lgbm_{datetime.utcnow():%Y%m%d_%H%M}.pkl"
    joblib.dump({"model": clf, "label_encoder": le}, model_path)
    print("saved model   ->", model_path)

    # ── 7. predict full dataset & persist ───────────────────────────────────
    proba = clf.predict_proba(embeddings)
    df["severity_pred"]     = le.inverse_transform(clf.predict(embeddings))
    df["severity_prob_max"] = proba.max(axis=1)

    out_path = PROC_DIR / f"master_plus_pred_{datetime.utcnow():%Y%m%d_%H%M}.parquet"
    df.to_parquet(out_path, index=False)
    print("wrote predictions ->", out_path)

def train():
    _train()

if __name__ == "__main__":
    _train()