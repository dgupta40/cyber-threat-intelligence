#!/usr/bin/env python3
"""
analyze_linked_cves.py — Post-linking analysis script

 Filters NVD rows with at least one linked article
 Saves them to a CSV for reporting or modeling
 Plots distribution of link counts for presentation
"""

import pandas as pd
import matplotlib.pyplot as plt


def analyze_linked_cves(
    input_csv="data/processed/master_linked.csv",
    output_csv="data/processed/linked_nvd_only.csv",
):
    print(f" Loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)

    # Filter NVD entries with n_articles > 0
    df_nvd = df[(df["source"] == "nvd")]
    df_linked = df_nvd[df_nvd["n_articles"].fillna(0).astype(int) > 0]
    print(f"Found {len(df_linked)} linked NVD records")

    # Save filtered rows
    df_linked.to_csv(output_csv, index=False)
    print(f" Saved linked CVEs to: {output_csv}")

    # Plot distribution of linked article counts
    dist = df_linked["n_articles"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    dist.plot(kind="bar", color="teal")
    plt.title("Distribution of Linked THN Articles per NVD CVE")
    plt.xlabel("Number of Linked THN Articles")
    plt.ylabel("Number of CVEs")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_linked_cves()
