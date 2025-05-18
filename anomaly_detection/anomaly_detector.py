# anomaly_detector.py – Isolation Forest on TF‑IDF vectors
"""
Flags text records whose content looks unusual compared to historical corpus.
Adds two columns to master.csv:
  • iso_score     – IsolationForest decision_function (higher = normal)
  • anomaly_flag  – True if iso_score < THRESH
This is a light, no‑GPU baseline suitable for an under‑grad capstone.
Usage
  python anomaly_detector.py
Dependencies
  pip install scikit-learn pandas joblib
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

ROOT  = Path(__file__).resolve().parent.parent
CSV   = ROOT / "data/processed/master.csv"
THRESH= -0.20   # anomaly threshold (tweak later)


def main():
    df = pd.read_csv(CSV)

    # quick drop NaNs -----------------------------------------------------------
    df = df.dropna(subset=["clean_text"])

    # vectorise all text -------------------------------------------------------
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english")
    X   = vec.fit_transform(df["clean_text"])  # sparse matrix

    # fit iso‑forest -----------------------------------------------------------
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)

    df["iso_score"]    = iso.decision_function(X)  # higher ~ normal
    df["anomaly_flag"] = df["iso_score"] < THRESH

    df.to_csv(CSV, index=False)
    print("[✓] anomaly_flag added to", CSV)

if __name__ == "__main__":
    main()
