#!/usr/bin/env python3
"""
validate_preprocessing.py - Validate CTI preprocessing results

This script performs key validation checks on the preprocessed CTI data
to ensure quality, consistency, and correctness before modeling.
"""

import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def run_validation(
    input_file="data/processed/master.csv", output_dir="validation_results"
):
    """Run key validation checks on preprocessed data"""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records: {df['source'].value_counts().to_dict()}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "file_analyzed": input_file,
        "record_count": len(df),
        "issues": [],
        "warnings": [],
        "statistics": {},
    }

    # 1. Basic data integrity checks
    print("\n1. Checking data integrity...")

    # 1.1 Check for null values in critical columns
    critical_columns = ["source", "clean_text", "published_dt"]
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            results["issues"].append(
                f"Missing data: {null_count} null values in '{col}'"
            )
            print(f" Found {null_count} null values in '{col}'")

    # 1.2 Check for empty clean_text
    empty_text = df["clean_text"].astype(str).str.strip().eq("").sum()
    if empty_text > 0:
        results["issues"].append(
            f"Empty text: {empty_text} records with empty 'clean_text'"
        )
        print(f" Found {empty_text} records with empty 'clean_text'")

    # 1.3 Verify source values
    invalid_sources = df[~df["source"].isin(["nvd", "thehackernews"])]
    if len(invalid_sources) > 0:
        results["issues"].append(
            f"Invalid sources: {len(invalid_sources)} records with unexpected source values"
        )
        print(f" Found {len(invalid_sources)} records with invalid sources")

    # 2. Text cleaning validation
    print("\n2. Validating text cleaning...")

    # 2.1 Sample records for text cleaning checks
    sample_size = min(500, len(df))
    sample = df.sample(sample_size)

    # 2.2 Check for HTML remnants
    html_count = sample["clean_text"].astype(str).str.contains("<[^>]+>").sum()
    html_rate = html_count / sample_size
    if html_rate > 0.01:  # More than 1% with HTML
        results["issues"].append(
            f"HTML remnants: {html_count} samples ({html_rate:.1%}) contain HTML tags"
        )
        print(f" {html_count} samples ({html_rate:.1%}) contain HTML tags")
    else:
        print(f" HTML cleaning: {html_rate:.1%} error rate")

    # 2.3 Check for CVE normalization
    cve_pattern = r"CVE-\d{4}-\d{4,7}"
    unnormalized = (
        sample["clean_text"].astype(str).str.contains(cve_pattern, case=False).sum()
    )
    unnorm_rate = unnormalized / sample_size
    if unnorm_rate > 0.05:  # More than 5% with unnormalized CVEs
        results["warnings"].append(
            f"CVE normalization: {unnormalized} samples ({unnorm_rate:.1%}) contain unnormalized CVE references"
        )
        print(
            f" {unnormalized} samples ({unnorm_rate:.1%}) contain unnormalized CVE references"
        )
    else:
        print(f" CVE normalization: {unnorm_rate:.1%} error rate")

    # 3. Feature validation
    print("\n3. Validating feature extraction...")

    # 3.1 Sentiment score range
    sentiment_min = df["sentiment"].min()
    sentiment_max = df["sentiment"].max()
    if sentiment_min < -1.0 or sentiment_max > 1.0:
        results["issues"].append(
            f"Sentiment range: found values outside [-1, 1] range: [{sentiment_min}, {sentiment_max}]"
        )
        print(
            f" Sentiment values outside expected range: [{sentiment_min}, {sentiment_max}]"
        )
    else:
        print(f" Sentiment range valid: [{sentiment_min:.2f}, {sentiment_max:.2f}]")

    # 3.2 NVD-specific checks
    nvd_df = df[df["source"] == "nvd"]
    if "cvss_score" in nvd_df.columns:
        invalid_cvss = nvd_df[
            (nvd_df["cvss_score"].notna())
            & ((nvd_df["cvss_score"] < 0) | (nvd_df["cvss_score"] > 10))
        ]
        if len(invalid_cvss) > 0:
            results["issues"].append(
                f"CVSS scores: {len(invalid_cvss)} records have values outside [0, 10] range"
            )
            print(f" {len(invalid_cvss)} CVSS scores outside valid range")
        else:
            print(f" CVSS scores within valid range")

    # 3.3 Check CVE extraction for NVD records
    if "primary_cve" in nvd_df.columns and "mentioned_cves" in nvd_df.columns:
        sample_nvd = nvd_df.sample(min(100, len(nvd_df)))
        cve_mismatch = 0

        for _, row in sample_nvd.iterrows():
            if pd.notna(row["primary_cve"]):
                mentioned = row.get("mentioned_cves", [])
                if isinstance(mentioned, str):
                    try:
                        mentioned = eval(
                            mentioned
                        )  # Convert string representation of list to actual list
                    except:
                        mentioned = []

                if row["primary_cve"] not in mentioned:
                    cve_mismatch += 1

        mismatch_rate = cve_mismatch / len(sample_nvd)
        if mismatch_rate > 0.05:  # More than 5% mismatches
            results["issues"].append(
                f"CVE extraction: {cve_mismatch} NVD records ({mismatch_rate:.1%}) missing primary CVE in mentioned_cves"
            )
            print(
                f" {cve_mismatch} NVD records ({mismatch_rate:.1%}) have CVE extraction issues"
            )
        else:
            print(f" CVE extraction: {mismatch_rate:.1%} error rate")

    # 4. Generate statistics
    print("\n4. Generating statistics...")

    # 4.1 Record counts by source
    source_counts = df["source"].value_counts().to_dict()
    results["statistics"]["source_counts"] = source_counts

    # 4.2 Temporal distribution
    if "year" in df.columns:
        year_counts = df["year"].value_counts().sort_index().to_dict()
        results["statistics"]["year_distribution"] = year_counts

    # 4.3 Sentiment statistics
    results["statistics"]["sentiment"] = {
        "min": float(df["sentiment"].min()),
        "max": float(df["sentiment"].max()),
        "mean": float(df["sentiment"].mean()),
        "median": float(df["sentiment"].median()),
    }

    # 4.4 CVSS statistics for NVD
    if "cvss_score" in nvd_df.columns:
        results["statistics"]["cvss"] = {
            "min": float(nvd_df["cvss_score"].min()),
            "max": float(nvd_df["cvss_score"].max()),
            "mean": float(nvd_df["cvss_score"].mean()),
            "null_percentage": float(nvd_df["cvss_score"].isnull().mean() * 100),
        }

    # 4.5 Feature presence
    for feature in ["has_exploit_mention", "has_patch_mention"]:
        if feature in df.columns:
            presence = df[feature].mean() * 100
            results["statistics"][f"{feature}_percentage"] = float(presence)

    # 5. Generate plots
    print("\n5. Creating visualization plots...")

    # 5.1 Setup plots
    plt.figure(figsize=(15, 10))

    # 5.2 Sentiment distribution
    plt.subplot(2, 2, 1)
    df["sentiment"].hist(bins=50)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")

    # 5.3 Records by year
    if "year" in df.columns:
        plt.subplot(2, 2, 2)
        year_series = df["year"].value_counts().sort_index()
        year_series.plot(kind="bar")
        plt.title("Records by Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

    # 5.4 CVSS score distribution
    if "cvss_score" in nvd_df.columns:
        plt.subplot(2, 2, 3)
        nvd_df["cvss_score"].hist(bins=20)
        plt.title("CVSS Score Distribution")
        plt.xlabel("CVSS Score")
        plt.ylabel("Count")

    # 5.5 Text length distribution
    plt.subplot(2, 2, 4)
    df["clean_text"].astype(str).str.len().hist(bins=50)
    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "validation_plots.png")
    plt.savefig(plot_path)
    plt.close()

    # 6. Output detailed sample for manual inspection
    print("\n6. Generating samples for manual inspection...")
    nvd_sample = df[df["source"] == "nvd"].sample(min(5, len(nvd_df)))
    thn_sample = df[df["source"] == "thehackernews"].sample(
        min(5, len(df[df["source"] == "thehackernews"]))
    )

    samples = pd.concat([nvd_sample, thn_sample])
    samples_path = os.path.join(output_dir, "samples_for_review.csv")
    samples.to_csv(samples_path, index=False)

    # 7. Save validation results
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # 8. Summary
    issue_count = len(results["issues"])
    warning_count = len(results["warnings"])

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total records: {len(df)}")
    print(f"Critical issues: {issue_count}")
    print(f"Warnings: {warning_count}")
    print(f"Results saved to {output_dir}/")

    if issue_count == 0:
        print("\n Preprocessing validation PASSED! No critical issues found.")
    else:
        print(f"\n Preprocessing validation found {issue_count} issues to resolve.")

    return results


if __name__ == "__main__":
    run_validation()
