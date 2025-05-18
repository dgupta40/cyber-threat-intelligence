# urgency_scoring.py – CVSS + sentiment → Low / Medium / High
"""
Adds two columns to master.csv:
  • urgency_score  – weighted score ∈ [0,1]
  • urgency_label  – Low / Medium / High (quantile cut‑offs)
Assumptions
  • master.csv has `cvss_score` (0‑10) and `sentiment` (‑1…1 or 0…1)
Usage
  python urgency_scoring.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data/processed/master.csv"

WEIGHT_CVSS      = 0.7   # 70 % technical risk
WEIGHT_SENTIMENT = 0.3   # 30 % public fear / buzz


def main():
    df = pd.read_csv(CSV)

    # normalise inputs ---------------------------------------------------------
    df["cvss_norm"] = df["cvss_score"].fillna(5.0).clip(0,10) / 10
    # if sentiment is in [‑1,1] range convert to [0,1]
    if (df["sentiment"].min() < 0):
        df["sentiment_norm"] = (df["sentiment"] + 1) / 2
    else:
        df["sentiment_norm"] = df["sentiment"].clip(0,1)

    # weighted urgency ---------------------------------------------------------
    df["urgency_score"] = (
        WEIGHT_CVSS      * df["cvss_norm"] +
        WEIGHT_SENTIMENT * df["sentiment_norm"]
    )

    # quantile thresholds: top 15 % High, next 25 % Medium --------------------
    q85, q60 = df["urgency_score"].quantile([0.85, 0.60])

    def label(u: float) -> str:
        if u >= q85: return "High"
        if u >= q60: return "Medium"
        return "Low"

    df["urgency_label"] = df["urgency_score"].apply(label)

    # persist ------------------------------------------------------------------
    df.to_csv(CSV, index=False)
    print("[✓] urgency_label added to", CSV)

if __name__ == "__main__":
    main()
