"""
Train a baseline severity-bin classifier.


Features  = SBERT embeddings  (768-d)
Target    = severity_bin  (critical / high / medium / low / unknown)
Model     = GradientBoostingClassifier  (sklearn)


Outputs
  • models/severity_gb.pkl          - fitted model
  • data/processed/master_<TS>.parquet gets two new cols:
        severity_pred, severity_prob_max
  • metrics/gb_report_<TS>.json     - precision / recall / f1 / support
"""


from __future__ import annotations
import json, joblib
from pathlib import Path
from datetime import datetime


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
METRICS_DIR= Path("metrics")
for p in (MODEL_DIR, METRICS_DIR): p.mkdir(exist_ok=True)


# ── 1.  Load latest parquet & SBERT ────────────────────────────────────────────
latest_parquet = max(PROC_DIR.glob("master_*.parquet"))
df   = pd.read_parquet(latest_parquet)
sbert_path = df["sbert_path"].iloc[0]      # same for all rows in run
X = np.load(sbert_path).astype("float32")


# ── 2.  Target encoding ───────────────────────────────────────────────────────
lbl = LabelEncoder()
mask = df["severity_bin"].ne("unknown")
df, X = df[mask], X[mask]
y   = lbl.fit_transform(df["severity_bin"].fillna("unknown"))


# optional: balance with class_weight
class_wts = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_wt_dict = {i: w for i, w in enumerate(class_wts)}


# ── 3.  Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=42
)


# ── 4.  Fit model ─────────────────────────────────────────────────────────────
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)


# ── 5.  Evaluate and save report ──────────────────────────────────────────────
y_pred = gb.predict(X_test)
report = classification_report(
    y_test, y_pred, target_names=lbl.classes_, output_dict=True
)
metrics_path = METRICS_DIR / f"gb_report_{datetime.utcnow():%Y%m%d_%H%M}.json"
with open(metrics_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved metrics -> {metrics_path}")


# ── 6.  Persist model ─────────────────────────────────────────────────────────
model_path = MODEL_DIR / f"severity_gb_{datetime.utcnow():%Y%m%d_%H%M}.pkl"
joblib.dump({"model": gb, "label_encoder": lbl}, model_path)
print(f"Saved model   -> {model_path}")


# ── 7.  Write predictions back to parquet for dashboard ───────────────────────
proba = gb.predict_proba(X)
df["severity_pred"]      = lbl.inverse_transform(gb.predict(X))
df["severity_prob_max"]  = proba.max(axis=1)


out_parquet = PROC_DIR / f"master_plus_pred_{datetime.utcnow():%Y%m%d_%H%M}.parquet"
df.to_parquet(out_parquet, index=False)
print(f"Wrote predictions -> {out_parquet}")



