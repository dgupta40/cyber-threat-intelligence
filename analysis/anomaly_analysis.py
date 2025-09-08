#!/usr/bin/env python3
"""
Analyze and visualize anomaly detection results from the CTI pipeline.
This script generates visualizations and metrics for the anomaly detection model.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import IsolationForest

# Configure logging and paths
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("anomaly_analysis")

DATA_FILE = Path("data/processed/emerging_threats.parquet")
OUTPUT_DIR = Path("analysis/anomaly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
def load_data():
    """Load the data with anomaly detection results."""
    log.info(f"Loading data from {DATA_FILE}")
    return pd.read_parquet(DATA_FILE)


# =============================================================
def generate_summary_stats(df):
    """Generate summary statistics for anomaly detection."""
    log.info("Generating summary statistics")
    emerg = df["emerging"].sum()
    total = len(df)
    pct = 100 * emerg / total

    # Individual method counts
    zc = df["zero_day_flag"].sum()
    sc = df["spike_flag"].sum()
    ifc = df["if_flag"].sum()

    # Overlaps
    only_z = ((df["zero_day_flag"]) & ~df["spike_flag"] & ~df["if_flag"]).sum()
    only_s = (~df["zero_day_flag"] & df["spike_flag"] & ~df["if_flag"]).sum()
    only_if = (~df["zero_day_flag"] & ~df["spike_flag"] & df["if_flag"]).sum()
    zs = (df["zero_day_flag"] & df["spike_flag"] & ~df["if_flag"]).sum()
    zi = (df["zero_day_flag"] & ~df["spike_flag"] & df["if_flag"]).sum()
    si = (~df["zero_day_flag"] & df["spike_flag"] & df["if_flag"]).sum()
    zsi = (df["zero_day_flag"] & df["spike_flag"] & df["if_flag"]).sum()

    print(f"\n{'='*50}")
    print("ANOMALY DETECTION SUMMARY")
    print(f"Total records: {total:,}, Emerging: {emerg:,} ({pct:.1f}%)")
    print(f"Zero-day: {zc:,}, Spike: {sc:,}, IF: {ifc:,}")
    print(
        f"Overlaps -> Z only: {only_z}, S only: {only_s}, IF only: {only_if}, Z+S: {zs}, Z+IF: {zi}, S+IF: {si}, All: {zsi}"
    )
    print(f"{'='*50}\n")

    # By month
    if "published_dt" in df:
        df["published_dt"] = pd.to_datetime(df["published_dt"], errors="ignore")
        monthly = df.groupby(df["published_dt"].dt.to_period("M"))["emerging"].agg(
            ["count", "sum"]
        )
        monthly["pct"] = 100 * monthly["sum"] / monthly["count"]
        print("Emerging by month:")
        print(monthly)

    return {
        "counts": [only_z, only_s, only_if, zs, zi, si, zsi],
        "method_totals": (zc, sc, ifc),
    }


# =============================================================
def plot_detection_methods_venn(stats, path):
    """Plot overlap using a Venn diagram."""
    try:
        from matplotlib_venn import venn3
    except ImportError:
        log.warning("matplotlib_venn not installed, skipping Venn plot")
        return

    a, b, c, ab, ac, bc, abc = stats["counts"]
    v = venn3(
        subsets=(a, b, ab, c, ac, bc, abc),
        set_labels=("Zero-day", "Spike", "IsolationForest"),
    )
    plt.title("Detection Method Overlap")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved Venn diagram to {path}")


# =============================================================
def plot_time_series(df, path):
    """Plot time series of emerging threats."""
    if "published_dt" not in df:
        return
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    weekly = (
        df.set_index("published_dt").resample("W")["emerging"].agg(["sum", "count"])
    )
    weekly["pct"] = 100 * weekly["sum"] / weekly["count"]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(weekly.index, weekly["count"], alpha=0.3, label="Total")
    ax1.bar(weekly.index, weekly["sum"], label="Emerging")
    ax1.set_ylabel("Count")
    ax2 = ax1.twinx()
    ax2.plot(weekly.index, weekly["pct"], color="green", label="% Emerging")
    ax2.set_ylabel("% Emerging")
    fig.legend(loc="upper left")
    plt.title("Weekly Emerging Threats")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved time series analysis to {path}")


# =============================================================
def visualize_high_dimensional_data(df, path):
    """t-SNE visualization of text embeddings."""
    log.info("Generating t-SNE visualization")
    sample = df.sample(min(1000, len(df)), random_state=42)
    vec = TfidfVectorizer(max_features=1000)
    X = vec.fit_transform(sample["clean_text"])
    rp = SparseRandomProjection(n_components=50, random_state=42)
    Xp = rp.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, init="random")
    Y = tsne.fit_transform(Xp)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        Y[:, 0], Y[:, 1], c=sample["emerging"].astype(int), cmap="coolwarm", alpha=0.6
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Emerging")
    ax.add_artist(legend1)
    ax.set_title("t-SNE of Emerging Threats")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved t-SNE visualization to {path}")


# =============================================================
def analyze_feature_importance(df, path):
    """Feature importance via simple RandomForest."""
    log.info("Analyzing feature importance")
    vec = CountVectorizer(max_features=100, stop_words="english")
    X = vec.fit_transform(df["clean_text"])
    y = df["emerging"]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    feat = vec.get_feature_names_out()
    imp = clf.feature_importances_
    top = sorted(zip(imp, feat), reverse=True)[:20]

    labels, values = zip(*[(f, i) for i, f in top])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, values)
    ax.set_title("Top Terms for Emerging Threats")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved feature importance to {path}")


# =============================================================
def evaluate_urgency_correlation(df, path):
    """Correlation between urgency score and emerging flag."""
    log.info("Analyzing urgency correlation")
    corr = df["urgency_score"].corr(df["emerging"])
    print(f"Urgency vs Emerging corr: {corr:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x="urgency_score", hue="emerging", bins=20, kde=True, ax=ax)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved urgency correlation plot to {path}")


# =============================================================
def analyze_category_distribution(df, path):
    """Analyze distribution of threat categories for emerging vs. non-emerging."""
    log.info("Analyzing category distribution")
    cats = [c for c in df.columns if c.startswith("category_")]
    if not cats:
        return
    records = []
    for c in cats:
        name = c.replace("category_", "")
        records.append(
            {"Category": name, "Type": "Emerging", "Count": df[c][df["emerging"]].sum()}
        )
        records.append(
            {
                "Category": name,
                "Type": "Non-Emerging",
                "Count": df[c][~df["emerging"]].sum(),
            }
        )
    dfa = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=dfa, x="Category", y="Count", hue="Type", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved category distribution to {path}")


# =============================================================
def main():
    log.info("Starting anomaly detection analysis")
    df = load_data()
    stats = generate_summary_stats(df)
    plot_detection_methods_venn(stats, OUTPUT_DIR / "venn.png")
    plot_time_series(df, OUTPUT_DIR / "time_series.png")
    visualize_high_dimensional_data(df, OUTPUT_DIR / "tsne.png")
    analyze_feature_importance(df, OUTPUT_DIR / "feature_importance.png")
    evaluate_urgency_correlation(df, OUTPUT_DIR / "urgency_corr.png")
    analyze_category_distribution(df, OUTPUT_DIR / "category_dist.png")
    log.info(f"Analysis complete. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
