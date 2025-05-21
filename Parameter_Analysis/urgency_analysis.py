#!/usr/bin/env python3
"""
Urgency Score Analysis & Weight Optimization (2025-05-18)

Analyzes and optimizes the urgency scoring system by:
1. Evaluating the current weighting scheme and distribution
2. Testing alternative weight combinations
3. Measuring sensitivity to each factor
4. Visualizing correlations and dependencies
5. Recommending optimal weights

Usage: python urgency_analysis.py
"""

import logging
import math
from pathlib import Path
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import ParameterGrid

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/master.parquet")
OUT_DIR = Path("data/analysis/urgency")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Current weights
CURRENT_WEIGHTS = {
    'severity':  0.35,  # CVSS-based
    'sentiment': 0.25,  # negative tone boosts urgency
    'exploit':   0.15,  # presence of exploit or PoC
    'patch':     0.15,  # absence of patch/fix
    'recency':   0.10,  # exponential decay over 30 days
    'articles':  0.05,  # log-scaled article count
}

# Weight variations to test
WEIGHT_OPTIONS = {
    'severity':  [0.25, 0.35, 0.45],
    'sentiment': [0.15, 0.25, 0.35],
    'exploit':   [0.05, 0.15, 0.25],
    'patch':     [0.05, 0.15, 0.25],
    'recency':   [0.05, 0.10, 0.15],
    'articles':  [0.00, 0.05, 0.10],
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_factor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and prepare all the factor data used for urgency calculation.
    Returns a DataFrame with normalized factor values.
    """
    now = datetime.utcnow()

    # Ensure published_date is datetime
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

    # Prepare all factors used for urgency calculation
    factors = pd.DataFrame(index=df.index)
    
    factors['severity'] = df['cvss_score'].fillna(0) / 10.0
    factors['sentiment'] = (df['sentiment'].fillna(0) + 1) / 2.0
    factors['exploit'] = df['clean_text'].str.contains(
        r'exploit|poc|proof of concept', case=False, na=False).astype(float)
    factors['patch'] = 1 - df['clean_text'].str.contains(
        r'patch|fix|update', case=False, na=False).astype(float)

    days = df['published_date'].apply(
        lambda d: (now - d).days if not pd.isna(d) else 365
    ).clip(lower=0)
    factors['days_old'] = days  # Keep raw days for analysis
    factors['recency'] = days.div(30).apply(lambda x: math.exp(-x))

    if 'n_articles' in df.columns:
        factors['n_articles_raw'] = df['n_articles'].fillna(0)  # Keep raw count for analysis
        factors['articles'] = factors['n_articles_raw'].apply(
            lambda x: math.log1p(x) / math.log1p(10)
        )
    else:
        factors['n_articles_raw'] = 0
        factors['articles'] = 0
        
    # Add text length as a potential new factor
    factors['text_length'] = df['clean_text'].str.len().fillna(0) / 1000  # Normalize to thousands
    
    return factors

def compute_score(factor_data: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute urgency score based on factors and specified weights.
    """
    score = sum(factor_data[factor] * weight for factor, weight in weights.items())
    return score

def compute_level(score: pd.Series) -> pd.Series:
    """
    Convert urgency score to urgency level.
    """
    level = pd.cut(
        score,
        bins=[0.0, 0.33, 0.66, 1.01],
        labels=['Low', 'Medium', 'High'],
        right=False
    )
    return level

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_factor_distributions(factor_data: pd.DataFrame):
    """
    Analyze and visualize the distribution of each factor.
    """
    logging.info("Analyzing factor distributions")
    
    # Select factors to analyze (exclude raw versions)
    analysis_factors = [f for f in factor_data.columns 
                       if f not in ['days_old', 'n_articles_raw']]
    
    # Plot histograms for all factors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, factor in enumerate(analysis_factors):
        ax = axes[i]
        sns.histplot(factor_data[factor], bins=30, ax=ax)
        ax.set_title(f'Distribution of {factor}')
        ax.set_xlabel(factor)
        ax.set_ylabel('Count')
        
        # Add statistics
        mean = factor_data[factor].mean()
        median = factor_data[factor].median()
        ax.text(0.7, 0.9, f'Mean: {mean:.3f}\nMedian: {median:.3f}', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'factor_distributions.png')
    plt.close()
    
    # Create a violin plot to compare distributions
    plt.figure(figsize=(12, 6))
    factor_melted = factor_data[analysis_factors].melt(var_name='Factor', value_name='Value')
    sns.violinplot(x='Factor', y='Value', data=factor_melted)
    plt.title('Comparison of Factor Distributions')
    plt.savefig(OUT_DIR / 'factor_comparison.png')
    plt.close()
    
    return {
        'means': factor_data[analysis_factors].mean(),
        'medians': factor_data[analysis_factors].median(),
        'std': factor_data[analysis_factors].std(),
        'min': factor_data[analysis_factors].min(),
        'max': factor_data[analysis_factors].max(),
    }

def analyze_factor_correlations(factor_data: pd.DataFrame):
    """
    Analyze and visualize correlations between factors.
    """
    logging.info("Analyzing factor correlations")
    
    # Compute correlation matrix
    corr_matrix = factor_data.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'factor_correlations.png')
    plt.close()
    
    return corr_matrix

def analyze_current_weights(factor_data: pd.DataFrame, weights: dict):
    """
    Analyze current weighting scheme and urgency distribution.
    """
    logging.info("Analyzing current weighting scheme")
    
    # Compute score using current weights
    urgency_score = compute_score(factor_data, weights)
    urgency_level = compute_level(urgency_score)
    
    # Analyze score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(urgency_score, bins=30, kde=True)
    plt.axvline(x=0.33, color='r', linestyle='--')
    plt.axvline(x=0.66, color='r', linestyle='--')
    plt.text(0.16, plt.ylim()[1]*0.9, 'Low', ha='center')
    plt.text(0.49, plt.ylim()[1]*0.9, 'Medium', ha='center')
    plt.text(0.83, plt.ylim()[1]*0.9, 'High', ha='center')
    plt.title('Distribution of Urgency Scores (Current Weights)')
    plt.xlabel('Urgency Score')
    plt.ylabel('Count')
    plt.savefig(OUT_DIR / 'current_score_distribution.png')
    plt.close()
    
    # Analyze level distribution
    level_counts = urgency_level.value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=level_counts.index, y=level_counts.values)
    plt.title('Distribution of Urgency Levels (Current Weights)')
    plt.xlabel('Urgency Level')
    plt.ylabel('Count')
    for i, count in enumerate(level_counts):
        plt.text(i, count + 5, str(count), ha='center')
    plt.savefig(OUT_DIR / 'current_level_distribution.png')
    plt.close()
    
    # Calculate contribution of each factor to final score
    contributions = {}
    for factor, weight in weights.items():
        contributions[factor] = factor_data[factor] * weight
    
    contributions_df = pd.DataFrame(contributions)
    contributions_df['total'] = urgency_score
    
    # Plot average contribution of each factor
    plt.figure(figsize=(10, 6))
    avg_contributions = contributions_df.mean()
    avg_contributions = avg_contributions.drop('total')
    colors = sns.color_palette('Set2', len(avg_contributions))
    
    # Sort by contribution
    avg_contributions = avg_contributions.sort_values(ascending=False)
    
    # Plot
    bars = plt.bar(avg_contributions.index, avg_contributions.values, color=colors)
    plt.title('Average Contribution of Each Factor to Urgency Score')
    plt.xlabel('Factor')
    plt.ylabel('Average Contribution')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'factor_contributions.png')
    plt.close()
    
    return {
        'score_stats': {
            'mean': urgency_score.mean(),
            'median': urgency_score.median(),
            'std': urgency_score.std(),
            'min': urgency_score.min(),
            'max': urgency_score.max(),
        },
        'level_counts': level_counts,
        'avg_contributions': avg_contributions,
    }

def perform_weight_sensitivity_analysis(factor_data: pd.DataFrame, base_weights: dict):
    """
    Perform sensitivity analysis on each weight parameter.
    """
    logging.info("Performing weight sensitivity analysis")
    
    # Baseline score
    baseline_score = compute_score(factor_data, base_weights)
    baseline_level = compute_level(baseline_score)
    baseline_level_counts = baseline_level.value_counts(normalize=True) * 100
    
    # Results storage
    sensitivity_results = {}
    
    # Test variations for each factor
    for factor in base_weights:
        results = []
        
        # Try different weights for this factor
        test_weights = np.linspace(0, 0.5, 11)  # 0.0 to 0.5 in 0.05 steps
        
        for test_weight in test_weights:
            # Create new weights with adjusted weight for current factor
            # and rescale others to ensure they sum to 1
            new_weights = base_weights.copy()
            new_weights[factor] = test_weight
            
            # Sum of all weights except the current factor
            remaining_weight = 1.0 - test_weight
            
            # Proportion of each remaining weight
            original_remaining_sum = sum(w for f, w in base_weights.items() if f != factor)
            if original_remaining_sum > 0:  # Avoid division by zero
                scaling_factor = remaining_weight / original_remaining_sum
                for f in new_weights:
                    if f != factor:
                        new_weights[f] = base_weights[f] * scaling_factor
            
            # Compute new score and level
            new_score = compute_score(factor_data, new_weights)
            new_level = compute_level(new_score)
            
            # Analyze changes
            score_change = (new_score - baseline_score).abs().mean()
            level_change_percent = (new_level != baseline_level).mean() * 100
            level_counts = new_level.value_counts(normalize=True) * 100
            
            # Store results
            results.append({
                'weight': test_weight,
                'score_change': score_change,
                'level_change_percent': level_change_percent,
                'low_percent': level_counts.get('Low', 0),
                'medium_percent': level_counts.get('Medium', 0),
                'high_percent': level_counts.get('High', 0),
            })
        
        sensitivity_results[factor] = pd.DataFrame(results)
    
    # Plot score sensitivity
    plt.figure(figsize=(12, 8))
    for factor, df in sensitivity_results.items():
        plt.plot(df['weight'], df['score_change'], marker='o', label=factor)
    
    plt.axvline(x=base_weights['severity'], color='gray', linestyle='--', label='Current severity')
    plt.axvline(x=base_weights['sentiment'], color='gray', linestyle='--')
    plt.axvline(x=base_weights['exploit'], color='gray', linestyle='--')
    
    plt.title('Sensitivity of Urgency Score to Weight Changes')
    plt.xlabel('Weight Value')
    plt.ylabel('Average Absolute Score Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / 'weight_sensitivity_score.png')
    plt.close()
    
    # Plot level change sensitivity
    plt.figure(figsize=(12, 8))
    for factor, df in sensitivity_results.items():
        plt.plot(df['weight'], df['level_change_percent'], marker='o', label=factor)
    
    plt.axvline(x=base_weights['severity'], color='gray', linestyle='--', label='Current severity')
    plt.axvline(x=base_weights['sentiment'], color='gray', linestyle='--')
    plt.axvline(x=base_weights['exploit'], color='gray', linestyle='--')
    
    plt.title('Percentage of Records Changing Urgency Level')
    plt.xlabel('Weight Value')
    plt.ylabel('Percentage of Changed Records')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / 'weight_sensitivity_level.png')
    plt.close()
    
    # Plot level distribution changes for the most sensitive factor
    most_sensitive = max(sensitivity_results.items(), 
                        key=lambda x: x[1]['level_change_percent'].max())[0]
    
    plt.figure(figsize=(12, 8))
    df = sensitivity_results[most_sensitive]
    plt.stackplot(df['weight'], 
                 df['low_percent'], 
                 df['medium_percent'],
                 df['high_percent'],
                 labels=['Low', 'Medium', 'High'],
                 colors=['green', 'orange', 'red'])
    
    plt.axvline(x=base_weights[most_sensitive], color='black', linestyle='--', 
               label=f'Current {most_sensitive} weight')
    
    plt.title(f'Impact of {most_sensitive} Weight on Urgency Level Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / f'{most_sensitive}_impact_on_levels.png')
    plt.close()
    
    return {
        'sensitivity_results': sensitivity_results,
        'most_sensitive_factor': most_sensitive,
    }

def test_weight_combinations(factor_data: pd.DataFrame):
    """
    Test different weight combinations and evaluate results.
    """
    logging.info("Testing weight combinations")
    
    # Generate valid weight combinations (ensuring sum = 1.0)
    def generate_valid_combinations():
        combinations = []
        
        # Get all possible weight options (filtering some combinations to reduce search space)
        # This approach ensures we have a manageable number of reasonable weight combinations
        param_grid = ParameterGrid({
            'severity': WEIGHT_OPTIONS['severity'],
            'sentiment': WEIGHT_OPTIONS['sentiment'],
            'exploit': WEIGHT_OPTIONS['exploit'],
            'patch': WEIGHT_OPTIONS['patch'],
            'recency': WEIGHT_OPTIONS['recency'],
            'articles': WEIGHT_OPTIONS['articles'],
        })
        
        for params in tqdm(param_grid, desc="Testing weight combinations", total=len(list(param_grid))):
            # Check if weights sum approximately to 1.0 (within rounding error)
            if 0.99 <= sum(params.values()) <= 1.01:
                # Normalize to exactly 1.0
                norm_factor = 1.0 / sum(params.values())
                normalized_params = {k: v * norm_factor for k, v in params.items()}
                combinations.append(normalized_params)
        
        return combinations
    
    # Get valid combinations
    valid_combinations = generate_valid_combinations()
    logging.info(f"Testing {len(valid_combinations)} valid weight combinations")
    
    # Define metrics to evaluate combinations
    def evaluate_combination(weights):
        # Compute score using weights
        score = compute_score(factor_data, weights)
        level = compute_level(score)
        
        # Calculate distribution metrics
        level_counts = level.value_counts(normalize=True)
        entropy = stats.entropy(level_counts)  # Higher entropy = more balanced distribution
        
        # Calculate variance in score
        score_variance = score.var()
        
        # Calculate proportion in each level
        low_pct = level_counts.get('Low', 0)
        med_pct = level_counts.get('Medium', 0)
        high_pct = level_counts.get('High', 0)
        
        # Calculate a balance score - the closer to equal distribution, the better
        # Perfect balance would be 1/3 for each level
        balance = 1 - (abs(low_pct - 1/3) + abs(med_pct - 1/3) + abs(high_pct - 1/3)) / 2
        
        return {
            'weights': weights,
            'entropy': entropy,
            'score_variance': score_variance,
            'balance': balance,
            'low_pct': low_pct,
            'med_pct': med_pct,
            'high_pct': high_pct,
        }
    
    # Evaluate all combinations
    results = []
    for weights in tqdm(valid_combinations, desc="Evaluating weights"):
        results.append(evaluate_combination(weights))
    
    results_df = pd.DataFrame(results)
    
    # Find best combinations based on different criteria
    best_entropy = results_df.loc[results_df['entropy'].idxmax()]
    best_variance = results_df.loc[results_df['score_variance'].idxmax()]
    best_balance = results_df.loc[results_df['balance'].idxmax()]
    
    # Also find a well-rounded option that scores reasonably well on all metrics
    results_df['combined_score'] = (
        (results_df['entropy'] / results_df['entropy'].max()) +
        (results_df['score_variance'] / results_df['score_variance'].max()) +
        (results_df['balance'] / results_df['balance'].max())
    ) / 3
    
    best_combined = results_df.loc[results_df['combined_score'].idxmax()]
    
    # Plot distribution for current weights and best alternative
    current_score = compute_score(factor_data, CURRENT_WEIGHTS)
    best_score = compute_score(factor_data, best_combined['weights'])
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(current_score, label='Current Weights', color='blue')
    sns.kdeplot(best_score, label='Optimized Weights', color='red')
    plt.axvline(x=0.33, color='gray', linestyle='--')
    plt.axvline(x=0.66, color='gray', linestyle='--')
    plt.title('Comparison of Urgency Score Distributions')
    plt.xlabel('Urgency Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(OUT_DIR / 'weight_optimization_comparison.png')
    plt.close()
    
    # Compare level distributions
    current_level = compute_level(current_score)
    best_level = compute_level(best_score)
    
    current_counts = current_level.value_counts().sort_index()
    best_counts = best_level.value_counts().sort_index()
    
    # Ensure all levels are represented
    for level in ['Low', 'Medium', 'High']:
        if level not in current_counts:
            current_counts[level] = 0
        if level not in best_counts:
            best_counts[level] = 0
    
    current_counts = current_counts.sort_index()
    best_counts = best_counts.sort_index()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(current_counts.index))
    
    plt.bar(x - width/2, current_counts.values, width, label='Current Weights')
    plt.bar(x + width/2, best_counts.values, width, label='Optimized Weights')
    
    plt.xlabel('Urgency Level')
    plt.ylabel('Count')
    plt.title('Comparison of Urgency Level Distributions')
    plt.xticks(x, current_counts.index)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(current_counts.values):
        plt.text(i - width/2, v + 5, str(int(v)), ha='center')
    
    for i, v in enumerate(best_counts.values):
        plt.text(i + width/2, v + 5, str(int(v)), ha='center')
    
    plt.savefig(OUT_DIR / 'level_distribution_comparison.png')
    plt.close()
    
    return {
        'all_results': results_df,
        'best_entropy': best_entropy,
        'best_variance': best_variance,
        'best_balance': best_balance,
        'best_combined': best_combined,
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("urgency_analysis")

    log.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    log.info(f"Loaded {len(df)} records")

    # Prepare factor data
    factor_data = prepare_factor_data(df)
    
    # Analyze factor distributions
    factor_stats = analyze_factor_distributions(factor_data)
    
    # Analyze factor correlations
    corr_matrix = analyze_factor_correlations(factor_data)
    
    # Analyze current weights
    current_analysis = analyze_current_weights(factor_data, CURRENT_WEIGHTS)
    
    # Perform sensitivity analysis
    sensitivity_results = perform_weight_sensitivity_analysis(factor_data, CURRENT_WEIGHTS)
    
    # Test different weight combinations
    optimization_results = test_weight_combinations(factor_data)
    
    # Prepare results summary
    best_weights = optimization_results['best_combined']['weights']
    
    # Save summary report
    with open(OUT_DIR / 'urgency_analysis_summary.txt', 'w') as f:
        f.write("# Urgency Score Analysis Summary\n\n")
        
        f.write("## Current Weights\n")
        for factor, weight in CURRENT_WEIGHTS.items():
            f.write(f"- {factor}: {weight:.2f}\n")
        f.write("\n")
        
        f.write("## Current Score Distribution\n")
        for stat, value in current_analysis['score_stats'].items():
            f.write(f"- {stat}: {value:.4f}\n")
        f.write("\n")
        
        f.write("## Current Level Distribution\n")
        for level, count in current_analysis['level_counts'].items():
            f.write(f"- {level}: {count} records ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## Factor Contributions\n")
        for factor, contribution in current_analysis['avg_contributions'].items():
            f.write(f"- {factor}: {contribution:.4f} ({contribution/current_analysis['score_stats']['mean']*100:.1f}% of average score)\n")
        f.write("\n")
        
        f.write("## Most Sensitive Factors\n")
        f.write(f"- Most sensitive factor: {sensitivity_results['most_sensitive_factor']}\n")
        f.write("\n")
        
        f.write("## Recommended Optimized Weights\n")
        for factor, weight in best_weights.items():
            f.write(f"- {factor}: {weight:.2f} (current: {CURRENT_WEIGHTS[factor]:.2f}, change: {weight-CURRENT_WEIGHTS[factor]:+.2f})\n")
        f.write("\n")
        
        f.write("## Optimization Metrics\n")
        f.write(f"- Entropy: {optimization_results['best_combined']['entropy']:.4f}\n")
        f.write(f"- Score Variance: {optimization_results['best_combined']['score_variance']:.4f}\n")
        f.write(f"- Balance Score: {optimization_results['best_combined']['balance']:.4f}\n")
        f.write("\n")
        
        f.write("## Level Distribution with Optimized Weights\n")
        f.write(f"- Low: {optimization_results['best_combined']['low_pct']*100:.1f}%\n")
        f.write(f"- Medium: {optimization_results['best_combined']['med_pct']*100:.1f}%\n")
        f.write(f"- High: {optimization_results['best_combined']['high_pct']*100:.1f}%\n")
        f.write("\n")
        
        f.write("## Correlation Analysis Findings\n")
        f.write("Top correlated factors:\n")
        # Get top 3 absolute correlations (excluding self-correlations)
        corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                             corr_matrix.iloc[i, j]))
        
        # Sort by absolute correlation and take top 3
        corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        for factor1, factor2, corr in corrs[:3]:
            f.write(f"- {factor1} and {factor2}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("## Suggested WEIGHTS Update\n")
        f.write("```python\n")
        f.write("WEIGHTS = {\n")
        for factor, weight in best_weights.items():
            f.write(f"    '{factor}': {weight:.2f},  # {'' if factor in CURRENT_WEIGHTS else 'NEW '}")
            if factor in CURRENT_WEIGHTS:
                f.write(f"was {CURRENT_WEIGHTS[factor]:.2f}")
            f.write("\n")
        f.write("}\n")
        f.write("```\n")
    
    log.info(f"Analysis complete. Results saved to {OUT_DIR}")
    log.info(f"Recommended weights: {best_weights}")

if __name__ == "__main__":
    main()