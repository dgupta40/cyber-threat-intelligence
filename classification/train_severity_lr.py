"""
Train baseline severity classifier
Features : TF-IDF (sparse)
Target   : severity_bin  (critical / high / medium / low)
Model    : LogisticRegression
Outputs  : models/severity_lr.pkl
           metrics/lr_report_<ts>.json
           data/processed/master_plus_pred_<ts>.parquet
"""

import glob, json, joblib, os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
METRICS_DIR= Path("metrics")
for p in (MODEL_DIR, METRICS_DIR): p.mkdir(exist_ok=True)

# ── 1. load latest parquet & TF‑IDF artefact ───────────────────────────────────
def _train() -> None:
    parquet = max(glob.glob(r"data/processed/master_*.parquet"))
    df      = pd.read_parquet(parquet)

    tfidf_pickle = max(glob.glob(r"models/tfidf_*.pkl"))
    tfidf_obj    = joblib.load(tfidf_pickle)    # {"model": vec, "matrix": X}
    X            = tfidf_obj["matrix"]

# ── 2. keep rows with a known label ────────────────────────────────────────────
    mask = df["severity_bin"].isin(["critical","high","medium","low"])
    df, X = df[mask], X[mask]

    le = LabelEncoder()
    y  = le.fit_transform(df["severity_bin"])

# ── 3. train/test split ───────────────────────────────────────────────────────
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )

# ── 4. fit fast LogisticRegression ────────────────────────────────────────────
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, ytr)

# ── 5. metrics ────────────────────────────────────────────────────────────────
    report = classification_report(yte, clf.predict(Xte),
                               target_names=le.classes_, output_dict=True)
    m_path = METRICS_DIR / f"lr_report_{datetime.utcnow():%Y%m%d_%H%M}.json"
    with open(m_path,"w") as f: json.dump(report, f, indent=2)
    print("saved metrics →", m_path)

# ── 6. save model ─────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / f"severity_lr_{datetime.utcnow():%Y%m%d_%H%M}.pkl"
    joblib.dump({"model": clf, "label_encoder": le, "vectorizer": tfidf_obj["model"]},
            model_path)
    print("saved model   →", model_path)

# ── 7. predict full set & persist ─────────────────────────────────────────────
    proba = clf.predict_proba(X)
    df["severity_pred"]      = le.inverse_transform(clf.predict(X))
    df["severity_prob_max"]  = proba.max(axis=1)

    out_parquet = PROC_DIR / f"master_plus_pred_{datetime.utcnow():%Y%m%d_%H%M}.parquet"
    df.to_parquet(out_parquet, index=False)
    print("wrote predictions →", out_parquet)

def train():
    _train()

if __name__ == "__main__":
    _train()