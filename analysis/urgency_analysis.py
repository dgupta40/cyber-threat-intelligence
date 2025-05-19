#!/usr/bin/env python3
"""
Analyze and visualize urgency scoring results.
Generates histograms, boxplots, and correlation statistics.
"""

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/urgency_assessed.parquet")
OUT_DIR   = Path("analysis/urgency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("urgency_analysis")

    log.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    # 1) distribution by level
    log.info("Plotting urgency‐level distribution")
    level_counts = df.urgency_level.value_counts(normalize=True).sort_index()
    plt.figure(figsize=(5,3))
    sns.barplot(x=level_counts.index, y=level_counts.values)
    plt.ylabel("Fraction of Records")
    plt.title("Urgency Level Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "level_distribution.png", dpi=300)
    plt.close()

    # 2) histogram of scores
    log.info("Plotting urgency score histogram")
    plt.figure(figsize=(6,4))
    sns.histplot(df.urgency_score, bins=30, kde=True)
    plt.xlabel("Urgency Score")
    plt.title("Distribution of Urgency Scores")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_histogram.png", dpi=300)
    plt.close()

    # 3) boxplot by level
    log.info("Plotting boxplot of scores by level")
    plt.figure(figsize=(6,4))
    sns.boxplot(x="urgency_level", y="urgency_score", data=df, order=["Low","Medium","High"])
    plt.title("Urgency Score by Level")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_boxplot.png", dpi=300)
    plt.close()

    # 4) correlation with emerging (if present)
    if "emerging" in df.columns:
        log.info("Plotting score vs emerging scatter")
        plt.figure(figsize=(5,4))
        sns.boxplot(x="emerging", y="urgency_score", data=df)
        plt.title("Urgency Score: Emerging vs. Non")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "emerging_boxplot.png", dpi=300)
        plt.close()
        corr = df.urgency_score.corr(df.emerging.astype(int))
        log.info(f"Correlation between urgency_score and emerging: {corr:.3f}")

    # 5) time‐series of average weekly score
    if "published_date" in df:
        log.info("Plotting weekly average urgency score")
        df["published_date"] = pd.to_datetime(df.published_date, errors="coerce")
        weekly = df.set_index("published_date").resample("W")["urgency_score"].mean()
        plt.figure(figsize=(8,3))
        weekly.plot()
        plt.ylabel("Avg Urgency Score")
        plt.title("Weekly Average Urgency Score Over Time")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "weekly_avg_score.png", dpi=300)
        plt.close()

    log.info(f"Analysis complete—outputs in {OUT_DIR}")

if __name__ == "__main__":
    main()
