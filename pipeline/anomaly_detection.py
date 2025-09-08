#!/usr/bin/env python3
"""
Detect emerging threats in the CTI pipeline (2025-05-18).

Methods:
  - Zero-day heuristic (regex on clean_text)
  - Mention spike detection via 30-day rolling z-score
  - Isolation Forest anomaly detection on sparse random projections of TF-IDF text

Outputs:
  - data/processed/emerging_threats.parquet with boolean 'emerging' flag
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/urgency_assessed.parquet")
OUT_FILE = Path("data/processed/emerging_threats.parquet")
ZERO_DAY_PATTERN = r"zero.?day|0.?day|unpatched"

# ──────────────────────────────────────────────────────────────────────────────
# CORE DETECTION LOGIC
# ──────────────────────────────────────────────────────────────────────────────


def detect_emerging(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add detection flags and final 'emerging' indicator to the DataFrame.
    """
    # 1) Zero-day heuristic flag
    df["zero_day_flag"] = df["clean_text"].str.contains(
        ZERO_DAY_PATTERN, case=False, na=False
    )

    # 2) Mention spike flag via 30-day rolling z-score
    df["published_dt"] = pd.to_datetime(df["published_date"], errors="coerce")
    daily_counts = df.groupby(df["published_dt"].dt.date).size()
    rolling_mean = daily_counts.rolling(window=30, min_periods=5).mean()
    rolling_std = daily_counts.rolling(window=30, min_periods=5).std()
    spikes = daily_counts[(daily_counts - rolling_mean) > 3 * rolling_std].index
    df["spike_flag"] = df["published_dt"].dt.date.isin(spikes)

    # 3) Isolation Forest on TF-IDF + random projection
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    X_text = tfidf.fit_transform(df["clean_text"])
    projector = SparseRandomProjection(n_components=256, random_state=42)
    X_proj = projector.fit_transform(X_text)
    iso = IsolationForest(contamination=0.03, random_state=42)
    df["if_flag"] = iso.fit_predict(X_proj) == -1

    # 4) Final emerging flag: any of the above
    df["emerging"] = df[["zero_day_flag", "spike_flag", "if_flag"]].any(axis=1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("anomaly_detection")

    log.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    log.info(f"Running emerging threat detection over {len(df)} rows")
    df_out = detect_emerging(df)

    log.info(f"Saving results to {OUT_FILE}")
    df_out.to_parquet(OUT_FILE, index=False)
    log.info(f"Saved emerging threats: {df_out['emerging'].sum()} flagged rows")


if __name__ == "__main__":
    main()
