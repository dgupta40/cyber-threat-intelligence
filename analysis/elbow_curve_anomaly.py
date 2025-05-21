#!/usr/bin/env python3
"""
Contamination parameter tuning for Isolation Forest in the CTI pipeline.
Generates an elbow curve to determine the optimal contamination value.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/urgency_assessed.parquet")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONTAMINATION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_contamination_parameter(df: pd.DataFrame):
    """
    Analyze different contamination values using elbow method.
    """
    # Prepare text data same way as in main pipeline
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    X_text = tfidf.fit_transform(df['clean_text'])
    projector = SparseRandomProjection(n_components=256, random_state=42)
    X_proj = projector.fit_transform(X_text)
    
    # Range of contamination values to test
    contamination_values = np.linspace(0.01, 0.1, 20)  # Test from 1% to 10% with finer granularity
    results = []
    
    log.info(f"Testing {len(contamination_values)} contamination values between {min(contamination_values):.2f} and {max(contamination_values):.2f}")
    
    # For each contamination value
    for contamination in contamination_values:
        log.info(f"Testing contamination={contamination:.4f}")
        # Fit the model
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X_proj)
        
        # Get anomaly scores
        scores = -iso.score_samples(X_proj)  # Negated so higher = more anomalous
        anomaly_labels = iso.predict(X_proj) == -1
        
        # Calculate metrics
        results.append({
            'contamination': contamination,
            'avg_anomaly_score': np.mean(scores),
            'anomalies_count': int(sum(anomaly_labels)),
            'score_variance': np.var(scores)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot the elbow curve
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average anomaly score
    plt.subplot(2, 2, 1)
    plt.plot(results_df['contamination'], results_df['avg_anomaly_score'], 'bo-')
    plt.xlabel('Contamination')
    plt.ylabel('Average Anomaly Score')
    plt.title('Average Anomaly Score vs Contamination')
    plt.grid(True)
    
    # Plot 2: Score variance
    plt.subplot(2, 2, 2)
    plt.plot(results_df['contamination'], results_df['score_variance'], 'ro-')
    plt.xlabel('Contamination')
    plt.ylabel('Variance of Anomaly Scores')
    plt.title('Score Variance vs Contamination')
    plt.grid(True)
    
    # Plot 3: Number of anomalies
    plt.subplot(2, 2, 3)
    plt.plot(results_df['contamination'], results_df['anomalies_count'], 'go-')
    plt.xlabel('Contamination')
    plt.ylabel('Number of Anomalies')
    plt.title('Number of Flagged Anomalies vs Contamination')
    plt.grid(True)
    
    # Plot 4: Derivative of avg anomaly score
    plt.subplot(2, 2, 4)
    score_diffs = np.gradient(results_df['avg_anomaly_score'])
    plt.plot(results_df['contamination'], score_diffs, 'mo-')
    plt.xlabel('Contamination')
    plt.ylabel('Rate of Change in Avg Score')
    plt.title('Derivative of Avg Score (Elbow Point Detection)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'contamination_elbow_analysis.png')
    
    # Find the elbow point using derivative method
    score_diffs = np.diff(results_df['avg_anomaly_score'])
    # Look for significant changes in the derivative
    second_derivs = np.diff(score_diffs)
    elbow_idx = np.argmax(np.abs(second_derivs)) + 1
    optimal_contamination = contamination_values[elbow_idx]
    
    # Alternative method: look for where the curve levels off (slope gets small)
    slope_threshold = 0.1 * max(abs(score_diffs))
    level_off_indices = np.where(abs(score_diffs) < slope_threshold)[0]
    if len(level_off_indices) > 0:
        alternative_idx = level_off_indices[0]
        alternative_contamination = contamination_values[alternative_idx]
    else:
        alternative_contamination = optimal_contamination
    
    log.info(f"Suggested optimal contamination (derivative method): {optimal_contamination:.4f}")
    log.info(f"Alternative contamination (leveling off): {alternative_contamination:.4f}")
    
    # Save results to CSV
    results_df.to_csv(OUTPUT_DIR / 'contamination_analysis_results.csv', index=False)
    
    return results_df, optimal_contamination

# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    global log
    log = logging.getLogger("contamination_tuning")

    log.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    
    log.info(f"Analyzing optimal contamination for {len(df)} documents")
    results_df, optimal_contamination = analyze_contamination_parameter(df)
    
    log.info(f"Analysis complete. Optimal contamination value: {optimal_contamination:.4f}")
    log.info(f"Visualization saved to {OUTPUT_DIR / 'contamination_elbow_analysis.png'}")
    log.info(f"Detailed results saved to {OUTPUT_DIR / 'contamination_analysis_results.csv'}")
    
    log.info("To use this value, update your detect_emerging() function with:")
    log.info(f"    iso = IsolationForest(contamination={optimal_contamination:.4f}, random_state=42)")

if __name__ == "__main__":
    main()