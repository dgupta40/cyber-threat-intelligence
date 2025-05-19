#!/usr/bin/env python3
"""
Compute urgency score & urgency level for each CVE/article (2025-05-18).
Factors: severity, sentiment, exploit presence, patch absence,
exponential recency decay, and number of linked articles.
"""

import logging
import math
from pathlib import Path
from datetime import datetime

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/master.parquet")
OUT_FILE  = Path("data/processed/urgency_assessed.parquet")

WEIGHTS = {
    'severity':  0.35,  # CVSS-based
    'sentiment': 0.25,  # negative tone boosts urgency
    'exploit':   0.15,  # presence of exploit or PoC
    'patch':     0.15,  # absence of patch/fix
    'recency':   0.10,  # exponential decay over 30 days
    'articles':  0.05,  # log-scaled article count
}

# ──────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns 'urgency_score' and 'urgency_level' to the DataFrame.
    """
    now = datetime.utcnow()

    # Ensure published_date is datetime
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

    severity = df['cvss_score'].fillna(0) / 10.0
    sentiment = (df['sentiment'].fillna(0) + 1) / 2.0
    exploit = df['clean_text'].str.contains(
        r'exploit|poc|proof of concept', case=False, na=False).astype(int)
    patch = 1 - df['clean_text'].str.contains(
        r'patch|fix|update', case=False, na=False).astype(int)

    days = df['published_date'].apply(
        lambda d: (now - d).days if not pd.isna(d) else 365
    ).clip(lower=0)
    recency = days.div(30).apply(lambda x: math.exp(-x))

    if 'n_articles' in df.columns:
        articles = df['n_articles'].fillna(0).apply(
            lambda x: math.log1p(x) / math.log1p(10)
        )
    else:
        articles = 0

    score = (
        severity * WEIGHTS['severity'] +
        sentiment * WEIGHTS['sentiment'] +
        exploit * WEIGHTS['exploit'] +
        patch * WEIGHTS['patch'] +
        recency * WEIGHTS['recency'] +
        articles * WEIGHTS['articles']
    )

    df = df.copy()
    df['urgency_score'] = score
    df['urgency_level'] = pd.cut(
        score,
        bins=[0.0, 0.33, 0.66, 1.01],
        labels=['Low', 'Medium', 'High'],
        right=False
    )
    return df

# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("urgency")

    log.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    log.info(f"Computing urgency scores for {len(df)} rows")
    df_out = compute_urgency(df)

    log.info(f"Saving results to {OUT_FILE}")
    df_out.to_parquet(OUT_FILE, index=False)
    df_out.to_csv(OUT_FILE.with_suffix('.csv'), index=False)
    log.info(f"Saved {len(df_out)} urgency scores to {OUT_FILE}")

if __name__ == "__main__":
    main()
