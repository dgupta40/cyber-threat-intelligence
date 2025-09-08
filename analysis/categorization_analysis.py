#!/usr/bin/env python3
"""
Evaluate threat‐type categorization performance and produce
a confusion matrix and F1‐score bar chart.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_FILE = Path("data/processed/master.parquet")
MODEL_FILE = Path("models/threat_model_with_sbert.pkl")
OUT_DIR = Path("analysis/categorization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct the exact labels used during training:
    - seed from CWE_MAP
    - rule-based patterns
    """
    from pipeline.threat_classifier import CATEGORY_PATTERNS, CWE_MAP, CATEGORIES

    def rule_labels(txt: str):
        out = []
        if isinstance(txt, str):
            t = txt.lower()
            for cat, pats in CATEGORY_PATTERNS.items():
                if any(__import__("re").search(p, t) for p in pats):
                    out.append(cat)
        return out

    y_labels = []
    for _, row in df.iterrows():
        seeds = []
        cwe = row.get("cwe")
        if pd.notna(cwe) and cwe in CWE_MAP:
            seeds.append(CWE_MAP[cwe])
        seeds += rule_labels(row.clean_text)
        y_labels.append(list(set(seeds)) or ["Other"])
    mlb = joblib.load(MODEL_FILE)["mlb"]
    return mlb.transform(y_labels), mlb.classes_


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("categorization_analysis")

    log.info("Loading data & model")
    df = pd.read_parquet(DATA_FILE)
    df = df[df.clean_text.str.strip().ne("")].reset_index(drop=True)
    mdl = joblib.load(MODEL_FILE)
    clf, tfidf, num_cols = mdl["model"], mdl["tfidf"], mdl["num_cols"]

    # 1) build features
    log.info("Building features")
    X_txt = tfidf.transform(df.clean_text)
    X_num = csr_matrix(df[num_cols].fillna(0).values)
    emb = np.load(Path("models") / "sbert_nvd.npy")
    emb = np.vstack([emb, np.load(Path("models") / "sbert_thn.npy")])
    X = hstack([X_txt, X_num, csr_matrix(emb)])

    # 2) labels
    y, categories = build_labels(df)

    # 3) train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4) predictions
    log.info("Predicting on test split")
    ypred = clf.predict(Xte)

    # 5) classification report
    report = classification_report(
        yte, ypred, target_names=categories, output_dict=True
    )
    rpt_df = pd.DataFrame(report).transpose()
    rpt_df.to_csv(OUT_DIR / "classification_report.csv")
    log.info(f"Saved classification report to {OUT_DIR/'classification_report.csv'}")

    # 6) confusion matrix
    log.info("Plotting confusion matrix")
    cm = confusion_matrix(yte.argmax(axis=1), ypred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(cm, display_labels=categories)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Threat Categorization Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    log.info(f"Saved confusion matrix to {OUT_DIR/'confusion_matrix.png'}")

    # 7) F1‐score bar chart
    log.info("Plotting F1‐scores")
    f1_scores = rpt_df.loc[categories, "f1-score"]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=f1_scores.values, y=categories)
    plt.xlabel("F1 Score")
    plt.xlim(0, 1)
    plt.title("F1‐score by Threat Category")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "f1_scores.png", dpi=300)
    plt.close()
    log.info(f"Saved F1‐score bar chart to {OUT_DIR/'f1_scores.png'}")


if __name__ == "__main__":
    main()
