#!/usr/bin/env python3
"""
Threat Classifier Performance Analysis & Parameter Optimization

This script analyzes and optimizes the multi-label threat classifier by:
1. Evaluating RandomForest parameter combinations
2. Analyzing feature importance for each category
3. Measuring contribution of different feature sets (TF-IDF, SBERT, numeric)
4. Visualizing confusion matrices and category correlations
"""

import logging
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import joblib

# ──────────────────────────────────────────────────────────────────────────────
DATA_FILE = Path("data/processed/master.parquet")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/analysis/threat_classifier")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Paths to the SBERT outputs
SBERT_NVD = MODEL_DIR / "sbert_nvd.npy"
SBERT_THN = MODEL_DIR / "sbert_thn.npy"

CATEGORIES = [
    "Phishing",
    "Ransomware",
    "Malware",
    "SQLInjection",
    "XSS",
    "DDoS",
    "ZeroDay",
    "SupplyChain",
    "Other",
]
CATEGORY_PATTERNS = {
    "Phishing": [r"phish", r"credential", r"email scam", r"spoof"],
    "Ransomware": [r"ransom", r"crypto.*currency", r"file.*locked"],
    "Malware": [r"malware", r"trojan", r"virus", r"worm"],
    "SQLInjection": [r"sql.*injection", r"database.*injection"],
    "XSS": [r"cross.?site.?script", r"xss"],
    "DDoS": [r"denial.?of.?service", r"ddos"],
    "ZeroDay": [r"zero.?day", r"0.?day", r"unpatched"],
    "SupplyChain": [r"supply.?chain", r"vendor", r"third.?party"],
}
CWE_MAP = {"CWE-79": "XSS", "CWE-89": "SQLInjection", "CWE-119": "Malware"}


def rule_labels(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    txt = text.lower()
    return [
        cat
        for cat, pats in CATEGORY_PATTERNS.items()
        if any(re.search(p, txt) for p in pats)
    ]


# ──────────────────────────────────────────────────────────────────────────────


def prepare_data(log):
    """Load and prepare the data, features, and labels"""
    log.info("Loading and preparing data")

    # 1) load data
    df = pd.read_parquet(DATA_FILE)
    df = df[df.clean_text.str.strip().ne("")]

    # 1b) drop null or all-whitespace clean_text rows
    df["clean_text"] = df["clean_text"].fillna("")
    df = df[df["clean_text"].str.strip().ne("")].reset_index(drop=True)

    # 2) build labels
    log.info("Generating labels")
    y_labels = []
    for _, row in df.iterrows():
        seeds = []
        cwe = row.get("cwe")
        if pd.notna(cwe) and cwe in CWE_MAP:
            seeds.append(CWE_MAP[cwe])
        seeds += rule_labels(row.clean_text)
        y_labels.append(list(set(seeds)) or ["Other"])

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y = mlb.fit_transform(y_labels)

    # Track label distributions for analysis
    label_counts = {
        cat: sum(1 for labels in y_labels if cat in labels) for cat in CATEGORIES
    }

    # Generate feature sets
    log.info("Generating features")

    # Text features
    tfidf = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.7, ngram_range=(1, 2))
    X_txt = tfidf.fit_transform(df.clean_text)

    # Numeric features
    num_cols = ["sentiment", "cvss_score"]
    if "n_articles" in df.columns:
        num_cols.append("n_articles")
    X_num = csr_matrix(df[num_cols].fillna(0).values)

    # SBERT embeddings
    log.info("Loading SBERT embeddings")
    emb_nvd = np.load(SBERT_NVD)
    emb_thn = np.load(SBERT_THN)
    # assumes master.parquet was built as [all NVD rows, then all THN rows]
    X_emb = csr_matrix(np.vstack([emb_nvd, emb_thn]))

    # Split the data
    all_features = {
        "TF-IDF": X_txt,
        "Numeric": X_num,
        "SBERT": X_emb,
        "All": hstack([X_txt, X_num, X_emb]),
    }

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        all_features["All"], y, test_size=0.2, random_state=42
    )

    # Create feature-specific train/test splits
    feature_splits = {}
    for feat_name, X in all_features.items():
        if feat_name != "All":  # Skip "All" as it's already split above
            X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            feature_splits[feat_name] = (X_tr, X_te)

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_splits": feature_splits,
        "all_features": all_features,
        "mlb": mlb,
        "label_counts": label_counts,
        "tfidf": tfidf,
        "num_cols": num_cols,
    }


def analyze_label_distribution(data, log):
    """Analyze and visualize the label distribution"""
    log.info("Analyzing label distribution")

    # Create a DataFrame for the label counts
    label_df = pd.DataFrame(
        {
            "Category": list(data["label_counts"].keys()),
            "Count": list(data["label_counts"].values()),
        }
    )

    # Sort by count descending
    label_df = label_df.sort_values("Count", ascending=False)

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Category", y="Count", data=label_df)
    plt.title("Distribution of Threat Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)

    # Add count annotations
    for i, count in enumerate(label_df["Count"]):
        ax.text(i, count + 5, str(count), ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png")

    # Calculate and plot label co-occurrence
    y_binary = data["mlb"].transform(
        data["df"].apply(
            lambda row: list(set(rule_labels(row["clean_text"]))) or ["Other"], axis=1
        )
    )

    co_occurrence = np.dot(y_binary.T, y_binary)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        co_occurrence,
        annot=True,
        fmt="d",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
    )
    plt.title("Category Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_cooccurrence.png")

    # Compute label correlations
    corr_matrix = np.corrcoef(y_binary.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
    )
    plt.title("Category Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_correlation.png")

    return {
        "label_df": label_df,
        "co_occurrence": co_occurrence,
        "correlation": corr_matrix,
    }


def tune_random_forest(data, log):
    """Tune RandomForest hyperparameters and evaluate performance"""
    log.info("Tuning RandomForest parameters")

    # Define parameter grid
    param_grid = {
        "estimator__n_estimators": [50, 80, 100, 120],
        "estimator__max_depth": [None, 10, 20, 30],
        "estimator__min_samples_split": [2, 5, 10],
    }

    # For faster tuning, use a subset of the data if needed
    X_sample = data["X_train"]
    y_sample = data["y_train"]

    # Base classifier
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

    # Multi-label classifier
    clf = OneVsRestClassifier(rf)

    # Grid search
    log.info("Starting GridSearchCV (this may take a while)")
    grid_search = GridSearchCV(
        clf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_sample, y_sample)
    elapsed_time = time.time() - start_time

    log.info(f"Grid search completed in {elapsed_time:.2f} seconds")
    log.info(f"Best parameters: {grid_search.best_params_}")
    log.info(f"Best score: {grid_search.best_score_:.4f}")

    # Train with best parameters and evaluate
    best_clf = grid_search.best_estimator_

    # Make predictions
    y_pred = best_clf.predict(data["X_test"])

    # Calculate per-class metrics
    results = classification_report(
        data["y_test"], y_pred, target_names=CATEGORIES, output_dict=True
    )

    # Save to CSV
    results_df = pd.DataFrame(results).transpose()
    results_df.to_csv(OUTPUT_DIR / "best_model_results.csv")

    # Plot results
    plot_classification_results(results, "Best Model Performance")

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_model": best_clf,
        "results": results,
        "y_pred": y_pred,
        "grid_search": grid_search,
    }


def analyze_feature_importance(data, best_model, log):
    """Analyze feature importance for each category"""
    log.info("Analyzing feature importance")

    # Extract feature importance from each classifier
    feature_importances = []

    # Get base models (each is a RandomForest for one class)
    base_models = best_model.estimators_

    # Feature names
    tfidf_features = data["tfidf"].get_feature_names_out()
    numeric_features = data["num_cols"]

    # Calculate SBERT feature count
    total_features = best_model.estimators_[0].feature_importances_.shape[0]
    sbert_features_count = total_features - len(tfidf_features) - len(numeric_features)

    # Create dummy feature names for SBERT
    sbert_features = [f"sbert_{i}" for i in range(sbert_features_count)]

    # Combined feature names
    all_feature_names = np.concatenate(
        [tfidf_features, numeric_features, sbert_features]
    )

    for i, model in enumerate(base_models):
        # Get feature importance
        importances = model.feature_importances_

        # Create sorted indices
        indices = np.argsort(importances)[::-1]

        # Get feature names
        feature_names = all_feature_names[indices]

        # Store data
        feature_importances.append(
            {
                "category": CATEGORIES[i],
                "importances": importances[indices],
                "feature_names": feature_names,
            }
        )

    # Summarize feature types importance for each category
    feature_type_importance = []
    for i, fi in enumerate(feature_importances):
        importances = fi["importances"]

        # Calculate importance by feature type
        tfidf_importance = sum(
            importances[
                np.array(
                    [
                        name.startswith("sbert_") == False
                        and name not in numeric_features
                        for name in fi["feature_names"]
                    ]
                )
            ]
        )
        numeric_importance = sum(
            importances[
                np.array([name in numeric_features for name in fi["feature_names"]])
            ]
        )
        sbert_importance = sum(
            importances[
                np.array([name.startswith("sbert_") for name in fi["feature_names"]])
            ]
        )

        feature_type_importance.append(
            {
                "category": CATEGORIES[i],
                "TF-IDF": tfidf_importance,
                "Numeric": numeric_importance,
                "SBERT": sbert_importance,
            }
        )

    # Convert to DataFrame for plotting
    type_importance_df = pd.DataFrame(feature_type_importance)
    type_importance_df = type_importance_df.set_index("category")

    # Plot feature type importance
    plt.figure(figsize=(12, 8))
    type_importance_df.plot(kind="bar", stacked=True, figsize=(12, 8))
    plt.title("Feature Type Importance by Category")
    plt.xlabel("Category")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_type_importance.png")

    # Plot top 10 features for each category
    for fi in feature_importances:
        category = fi["category"]
        importances = fi["importances"][:10]  # Top 10
        features = fi["feature_names"][:10]  # Top 10

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances, align="center")
        plt.yticks(range(len(importances)), features)
        plt.title(f"Top 10 Features for {category}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"top_features_{category}.png")
        plt.close()

    return {
        "feature_importances": feature_importances,
        "feature_type_importance": feature_type_importance,
        "feature_type_df": type_importance_df,
    }


def compare_feature_sets(data, log):
    """Compare performance using different feature sets"""
    log.info("Comparing performance of different feature sets")

    feature_combinations = {
        "TF-IDF Only": ["TF-IDF"],
        "SBERT Only": ["SBERT"],
        "TF-IDF + Numeric": ["TF-IDF", "Numeric"],
        "SBERT + Numeric": ["SBERT", "Numeric"],
        "All Features": ["TF-IDF", "Numeric", "SBERT"],
    }

    results = {}

    for combo_name, features in feature_combinations.items():
        log.info(f"Evaluating {combo_name}")

        # Create combined feature matrix
        features_to_combine = []
        for feat in features:
            X_tr, X_te = data["feature_splits"].get(feat, (None, None))
            if X_tr is not None and X_te is not None:
                features_to_combine.append((X_tr, X_te))

        if not features_to_combine:
            continue

        # Combine features
        X_train_combo = hstack([X_tr for X_tr, _ in features_to_combine])
        X_test_combo = hstack([X_te for _, X_te in features_to_combine])

        # Train model with default parameters for comparison
        rf = RandomForestClassifier(
            n_estimators=80, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf = OneVsRestClassifier(rf, n_jobs=-1)

        # Train
        start_time = time.time()
        clf.fit(X_train_combo, data["y_train"])
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        y_pred = clf.predict(X_test_combo)
        predict_time = time.time() - start_time

        # Evaluate
        report = classification_report(
            data["y_test"], y_pred, target_names=CATEGORIES, output_dict=True
        )

        # Store results
        results[combo_name] = {
            "model": clf,
            "report": report,
            "train_time": train_time,
            "predict_time": predict_time,
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }

    # Create performance comparison
    performance_df = pd.DataFrame(
        {
            "Feature Set": list(results.keys()),
            "F1 Macro": [r["f1_macro"] for r in results.values()],
            "F1 Weighted": [r["f1_weighted"] for r in results.values()],
            "Training Time (s)": [r["train_time"] for r in results.values()],
            "Prediction Time (s)": [r["predict_time"] for r in results.values()],
        }
    )

    # Save to CSV
    performance_df.to_csv(OUTPUT_DIR / "feature_set_comparison.csv", index=False)

    # Plot F1 scores
    plt.figure(figsize=(12, 6))

    x = np.arange(len(performance_df))
    width = 0.35

    plt.bar(x - width / 2, performance_df["F1 Macro"], width, label="F1 Macro")
    plt.bar(x + width / 2, performance_df["F1 Weighted"], width, label="F1 Weighted")

    plt.xlabel("Feature Set")
    plt.ylabel("F1 Score")
    plt.title("Classification Performance by Feature Set")
    plt.xticks(x, performance_df["Feature Set"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_set_f1_comparison.png")

    # Plot training and prediction times
    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2, performance_df["Training Time (s)"], width, label="Training Time"
    )
    plt.bar(
        x + width / 2,
        performance_df["Prediction Time (s)"],
        width,
        label="Prediction Time",
    )

    plt.xlabel("Feature Set")
    plt.ylabel("Time (s)")
    plt.title("Processing Time by Feature Set")
    plt.xticks(x, performance_df["Feature Set"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_set_time_comparison.png")

    return {"results": results, "performance_df": performance_df}


def analyze_multilabel_confusion(y_true, y_pred, categories, log):
    """Analyze and visualize multilabel confusion matrices"""
    log.info("Analyzing multilabel confusion matrices")

    # Calculate the multilabel confusion matrix
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    # For each category, plot the confusion matrix
    for i, category in enumerate(categories):
        cm = conf_matrices[i]

        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not " + category, category],
            yticklabels=["Not " + category, category],
        )
        plt.title(f"Confusion Matrix for {category}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Add metrics as text
        plt.figtext(
            0.5,
            0.01,
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}",
            ha="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"confusion_matrix_{category}.png")
        plt.close()

    return conf_matrices


def plot_classification_results(results, title):
    """Plot the classification report results"""
    # Extract metrics for each class
    categories = list(results.keys())[:-3]  # Exclude avg rows
    precision = [results[cat]["precision"] for cat in categories]
    recall = [results[cat]["recall"] for cat in categories]
    f1 = [results[cat]["f1-score"] for cat in categories]

    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.25

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{title.lower().replace(' ', '_')}.png")
    plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("threat_classifier_analysis")

    log.info("Starting threat classifier performance analysis")

    # Prepare data
    data = prepare_data(log)

    # Analyze label distribution
    label_analysis = analyze_label_distribution(data, log)

    # Compare feature sets
    feature_comparison = compare_feature_sets(data, log)

    # Tune RandomForest parameters
    tuning_results = tune_random_forest(data, log)

    # Analyze feature importance
    importance_analysis = analyze_feature_importance(
        data, tuning_results["best_model"], log
    )

    # Analyze confusion matrices
    confusion_analysis = analyze_multilabel_confusion(
        data["y_test"], tuning_results["y_pred"], CATEGORIES, log
    )

    # Save final model with optimal parameters
    log.info("Saving optimized model")
    joblib.dump(
        {
            "model": tuning_results["best_model"],
            "tfidf": data["tfidf"],
            "mlb": data["mlb"],
            "num_cols": data["num_cols"],
            "feature_importance": importance_analysis["feature_type_importance"],
            "best_params": tuning_results["best_params"],
        },
        OUTPUT_DIR / "optimized_threat_model.pkl",
    )

    # Generate summary report
    summary_file = OUTPUT_DIR / "analysis_summary.txt"
    with open(summary_file, "w") as f:
        f.write("# Threat Classifier Analysis Summary\n\n")

        f.write("## Dataset Information\n")
        f.write(f"- Total samples: {len(data['df'])}\n")
        f.write(f"- Feature dimensions: {data['X_train'].shape[1]}\n\n")

        f.write("## Label Distribution\n")
        for category, count in data["label_counts"].items():
            f.write(f"- {category}: {count} samples\n")
        f.write("\n")

        f.write("## Best Model Parameters\n")
        for param, value in tuning_results["best_params"].items():
            f.write(f"- {param}: {value}\n")
        f.write(f"- Best CV Score (F1 Macro): {tuning_results['best_score']:.4f}\n\n")

        f.write("## Feature Set Comparison\n")
        f.write(feature_comparison["performance_df"].to_string(index=False))
        f.write("\n\n")

        f.write("## Feature Type Importance\n")
        f.write("Average importance by feature type:\n")
        avg_importance = importance_analysis["feature_type_df"].mean()
        for feat_type, importance in avg_importance.items():
            f.write(f"- {feat_type}: {importance:.4f}\n")
        f.write("\n")

        f.write("## Performance Summary\n")
        f.write("Macro Average Metrics:\n")
        macro_metrics = tuning_results["results"]["macro avg"]
        for metric, value in macro_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")

        f.write("Per-Category F1 Scores:\n")
        for category in CATEGORIES:
            f1 = tuning_results["results"][category]["f1-score"]
            f.write(f"- {category}: {f1:.4f}\n")

    log.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    log.info(f"Summary report generated at {summary_file}")


if __name__ == "__main__":
    main()
