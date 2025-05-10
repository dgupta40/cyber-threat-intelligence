#!/usr/bin/env python3
"""
clean_eda_basic.py
─────────────────
Quick EDA on preprocessed master parquet:
  1. Missing-value % by column
  2. Token-count distribution
  3. CVSS score histogram (with missing shown)
  4. Severity-bin counts

Usage:
  pip install pandas matplotlib
  python clean_eda_basic.py --proc <path_to_master.parquet> --out charts/clean
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.autolayout": True})

def main(proc_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(proc_path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]:,} cols")

    # 1. Missingness
    pct_missing = df.isna().mean() * 100
    pct_missing.sort_values().plot(
        kind="barh",
        figsize=(6, len(pct_missing) * 0.25),
        title="Missing-value % by column"
    )
    plt.xlabel("Percent missing")
    plt.savefig(out_dir / "missingness.png")
    plt.close()

    # 2. Token count per document
    df["token_count"] = df["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["token_count"].plot(
        kind="hist",
        bins=30,
        figsize=(8,4),
        title="Token count per document"
    )
    plt.xlabel("Number of tokens")
    plt.savefig(out_dir / "token_count_hist.png")
    plt.close()

    # 3. CVSS score distribution
    scores = df["cvssScore"].fillna(-1)
    scores.plot(
        kind="hist",
        bins=40,
        figsize=(8,4),
        title="CVSS Score Distribution (–1 = missing)"
    )
    plt.xlabel("CVSS score")
    plt.savefig(out_dir / "cvss_score_hist.png")
    plt.close()

    # 4. Severity bins
    df["severity_bin"].value_counts().sort_index().plot(
        kind="bar",
        figsize=(6,4),
        title="Severity bin counts"
    )
    plt.ylabel("Count")
    plt.savefig(out_dir / "severity_bins.png")
    plt.close()

    print(f"✓ Clean-data EDA charts saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proc", required=True, type=Path,
        help="Path to cleaned master_*.parquet"
    )
    parser.add_argument(
        "--out", default=Path("charts/clean"), type=Path,
        help="Output directory for charts"
    )
    args = parser.parse_args()
    main(args.proc, args.out)
