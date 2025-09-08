#!/usr/bin/env python3
"""
TF-IDF Parameter Optimization for CTI Pipeline

Evaluates different min_df and max_df settings to find optimal values by analyzing:
- Vocabulary size impact
- Term coverage
- Information retention
- Sparsity
- Most important terms at different settings

Output: Visualizations and metrics for parameter selection justification
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json

# Configuration
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the processed data from master.parquet"""
    df = pd.read_parquet(DATA_DIR / "master.parquet")
    # Filter to only THN articles since that's what your original script uses for TF-IDF
    df_thn = df[df.source == "thehackernews"].copy()
    return df_thn


def analyze_tfidf_params(df):
    """Analyze impact of different min_df and max_df parameters"""
    # Document counts for reference
    n_docs = len(df)
    logging.info(f"Analyzing TF-IDF parameters on {n_docs} documents")

    # Parameter grids to test
    min_df_values = [1, 2, 3, 5, 10]
    max_df_values = [0.7, 0.8, 0.9, 0.95, 1.0]

    # Collect metrics
    results = []

    # Get raw term counts for reference
    all_words = " ".join(df.clean_text).split()
    total_words = len(all_words)
    unique_words = len(set(all_words))
    word_freq = Counter(all_words)
    logging.info(f"Total words: {total_words}, Unique words: {unique_words}")

    # Save most common words for reference
    most_common = pd.DataFrame(word_freq.most_common(50), columns=["term", "count"])
    most_common["document_percent"] = most_common["count"] / n_docs * 100
    most_common.to_csv(OUTPUT_DIR / "most_common_terms.csv", index=False)

    # Create reference vectorization with minimal filtering
    reference_tfidf = TfidfVectorizer(min_df=1, max_df=1.0)
    reference_matrix = reference_tfidf.fit_transform(df.clean_text)
    reference_vocab_size = len(reference_tfidf.vocabulary_)

    # Test different parameter combinations
    for min_df in min_df_values:
        for max_df in max_df_values:
            logging.info(f"Testing min_df={min_df}, max_df={max_df}")

            # Create and fit the vectorizer
            tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)
            X = tfidf.fit_transform(df.clean_text)

            # Calculate metrics
            vocab_size = len(tfidf.vocabulary_)
            sparsity = (1.0 - (X.nnz / float(X.shape[0] * X.shape[1]))) * 100
            vocab_coverage = vocab_size / unique_words * 100
            retained_terms = vocab_size
            filtered_low_freq = sum(
                1 for term, count in word_freq.items() if count < min_df
            )

            # Document frequency analysis
            df_counts = np.bincount(X.nonzero()[1], minlength=X.shape[1])
            high_df_terms = sum(df_counts >= max_df * n_docs)

            # Get top terms by TF-IDF score
            feature_names = tfidf.get_feature_names_out()
            if len(feature_names) > 0:
                # Get average TF-IDF score for each term
                tfidf_means = X.mean(axis=0).A1
                top_term_indices = tfidf_means.argsort()[-20:][::-1]
                top_terms = [feature_names[i] for i in top_term_indices]
            else:
                top_terms = []

            # Store results
            results.append(
                {
                    "min_df": min_df,
                    "max_df": max_df,
                    "vocab_size": vocab_size,
                    "vocab_coverage_pct": vocab_coverage,
                    "sparsity_pct": sparsity,
                    "removed_low_freq": filtered_low_freq,
                    "removed_high_freq": high_df_terms,
                    "top_terms": top_terms[:10],  # Save top 10 terms
                    "vocab_percent_of_reference": (
                        vocab_size / reference_vocab_size * 100
                        if reference_vocab_size > 0
                        else 0
                    ),
                }
            )

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "tfidf_parameter_analysis.csv", index=False)

    # Create visualizations

    # 1. Vocabulary size heatmap
    plot_heatmap(
        results_df,
        min_df_values,
        max_df_values,
        "vocab_size",
        "Vocabulary Size by TF-IDF Parameters",
        "vocabulary_size_heatmap.png",
    )

    # 2. Sparsity heatmap
    plot_heatmap(
        results_df,
        min_df_values,
        max_df_values,
        "sparsity_pct",
        "Matrix Sparsity (%) by TF-IDF Parameters",
        "sparsity_heatmap.png",
    )

    # 3. Vocabulary coverage
    plot_heatmap(
        results_df,
        min_df_values,
        max_df_values,
        "vocab_coverage_pct",
        "Vocabulary Coverage (%) by TF-IDF Parameters",
        "coverage_heatmap.png",
    )

    # 4. Plot percent of reference vocabulary
    plot_heatmap(
        results_df,
        min_df_values,
        max_df_values,
        "vocab_percent_of_reference",
        "Percentage of Reference Vocabulary Retained",
        "reference_vocab_heatmap.png",
    )

    # Create visual summary for elbow curve
    # We'll use min_df=3 as fixed (from your original) and see impact of max_df
    fixed_min_df = 3
    fixed_min_df_results = results_df[results_df.min_df == fixed_min_df].sort_values(
        "max_df"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vocabulary size by max_df (with min_df=3)
    axes[0].plot(
        fixed_min_df_results["max_df"], fixed_min_df_results["vocab_size"], "bo-"
    )
    axes[0].set_xlabel("max_df")
    axes[0].set_ylabel("Vocabulary Size")
    axes[0].set_title(f"Vocabulary Size (min_df={fixed_min_df})")
    axes[0].grid(True)

    # Sparsity by max_df (with min_df=3)
    axes[1].plot(
        fixed_min_df_results["max_df"], fixed_min_df_results["sparsity_pct"], "ro-"
    )
    axes[1].set_xlabel("max_df")
    axes[1].set_ylabel("Sparsity (%)")
    axes[1].set_title(f"Matrix Sparsity (min_df={fixed_min_df})")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "max_df_impact.png")

    # Now fixed max_df=0.8 and varying min_df
    fixed_max_df = 0.8
    fixed_max_df_results = results_df[results_df.max_df == fixed_max_df].sort_values(
        "min_df"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vocabulary size by min_df (with max_df=0.8)
    axes[0].plot(
        fixed_max_df_results["min_df"], fixed_max_df_results["vocab_size"], "bo-"
    )
    axes[0].set_xlabel("min_df")
    axes[0].set_ylabel("Vocabulary Size")
    axes[0].set_title(f"Vocabulary Size (max_df={fixed_max_df})")
    axes[0].grid(True)

    # Sparsity by min_df (with max_df=0.8)
    axes[1].plot(
        fixed_max_df_results["min_df"], fixed_max_df_results["sparsity_pct"], "ro-"
    )
    axes[1].set_xlabel("min_df")
    axes[1].set_ylabel("Sparsity (%)")
    axes[1].set_title(f"Matrix Sparsity (max_df={fixed_max_df})")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "min_df_impact.png")

    # Create recommendation
    optimal_row = find_optimal_params(results_df)

    # Save recommendations
    with open(OUTPUT_DIR / "tfidf_recommendations.txt", "w") as f:
        f.write("TF-IDF Parameter Analysis Results\n")
        f.write("=================================\n\n")
        f.write(f"Dataset: {n_docs} documents, {unique_words} unique terms\n\n")

        f.write("Recommended Parameters:\n")
        f.write(f"min_df={optimal_row['min_df']}, max_df={optimal_row['max_df']}\n\n")

        f.write("Justification:\n")
        f.write(f"- Vocabulary size: {optimal_row['vocab_size']} terms\n")
        f.write(
            f"- Vocabulary coverage: {optimal_row['vocab_coverage_pct']:.1f}% of unique terms\n"
        )
        f.write(f"- Matrix sparsity: {optimal_row['sparsity_pct']:.1f}%\n")
        f.write(f"- Removed {optimal_row['removed_low_freq']} low-frequency terms\n")
        f.write(
            f"- Removed {optimal_row['removed_high_freq']} high-frequency terms\n\n"
        )

        f.write("Top terms by TF-IDF importance with these parameters:\n")
        for i, term in enumerate(optimal_row["top_terms"], 1):
            f.write(f"{i}. {term}\n")

    return results_df


def plot_heatmap(df, x_values, y_values, metric, title, filename):
    """Create a heatmap visualization for a given metric"""
    pivot_table = df.pivot_table(index="min_df", columns="max_df", values=metric)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def find_optimal_params(results_df):
    """
    Find optimal parameters based on a scoring function
    This is a simplified approach - you can customize the scoring based on your needs
    """
    # Create a score combining vocabulary size, coverage, and reasonable sparsity
    # Higher score is better
    results_df["score"] = (
        results_df["vocab_coverage_pct"] * 0.4  # We want good coverage
        + (100 - results_df["sparsity_pct"])
        * 0.3  # Less sparsity is better (to a point)
        + results_df["vocab_percent_of_reference"] * 0.3  # Retain important terms
    )

    # Find the row with the highest score
    optimal_row = results_df.loc[results_df["score"].idxmax()]
    return optimal_row


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("Starting TF-IDF parameter analysis")

    # Load the THN data
    df_thn = load_data()

    # Run the analysis
    results = analyze_tfidf_params(df_thn)

    logging.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
