#!/usr/bin/env python3
"""
Threat-type multi-label classifier (2025-05-18) with SBERT features included.
Uses TF-IDF + SBERT embeddings + numeric features (sentiment, CVSS, n_articles).
Parallel RandomForest for speed.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ──────────────────────────────────────────────────────────────────────────────
DATA_FILE   = Path("data/processed/master.parquet")
MODEL_DIR   = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# Paths to the SBERT outputs you already saved
SBERT_NVD   = MODEL_DIR / "sbert_nvd.npy"
SBERT_THN   = MODEL_DIR / "sbert_thn.npy"

CATEGORIES  = ["Phishing","Ransomware","Malware","SQLInjection","XSS",
               "DDoS","ZeroDay","SupplyChain","Other"]
CATEGORY_PATTERNS = {
    "Phishing":      [r"phish", r"credential", r"email scam", r"spoof"],
    "Ransomware":    [r"ransom", r"crypto.*currency", r"file.*locked"],
    "Malware":       [r"malware", r"trojan", r"virus", r"worm"],
    "SQLInjection":  [r"sql.*injection", r"database.*injection"],
    "XSS":           [r"cross.?site.?script", r"xss"],
    "DDoS":          [r"denial.?of.?service", r"ddos"],
    "ZeroDay":       [r"zero.?day", r"0.?day", r"unpatched"],
    "SupplyChain":   [r"supply.?chain", r"vendor", r"third.?party"]
}
CWE_MAP = {"CWE-79":"XSS","CWE-89":"SQLInjection","CWE-119":"Malware"}

def rule_labels(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    txt = text.lower()
    return [cat for cat,pats in CATEGORY_PATTERNS.items() if any(re.search(p, txt) for p in pats)]

# ──────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("threat_classifier")

    # 1) load data
    df = pd.read_parquet(DATA_FILE)
    df = df[df.clean_text.str.strip().ne("")]

    # 1b) drop null or all-whitespace clean_text rows
    df["clean_text"] = df["clean_text"].fillna("")
    df = df[df["clean_text"].str.strip().ne("")].reset_index(drop=True)

    # 2) build labels
    y_labels = []
    for _, row in df.iterrows():
        seeds = []
        cwe = row.get("cwe")
        if pd.notna(cwe) and cwe in CWE_MAP:
            seeds.append(CWE_MAP[cwe])
        seeds += rule_labels(row.clean_text)
        y_labels.append(list(set(seeds)) or ["Other"])
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y   = mlb.fit_transform(y_labels)

    # 3) text features
    tfidf = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.7, ngram_range=(1,2))
    X_txt = tfidf.fit_transform(df.clean_text)

    # 4) numeric features
    num_cols = ["sentiment","cvss_score"]
    if "n_articles" in df.columns:
        num_cols.append("n_articles")
    X_num = csr_matrix(df[num_cols].fillna(0).values)

    # 5) SBERT embeddings
    log.info("Loading SBERT embeddings")
    emb_nvd = np.load(SBERT_NVD)
    emb_thn = np.load(SBERT_THN)
    # assumes master.parquet was built as [all NVD rows, then all THN rows]
    X_emb = csr_matrix(np.vstack([emb_nvd, emb_thn]))

    # 6) combine everything
    X = hstack([X_txt, X_num, X_emb])

    # 7) split, train, evaluate
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info("Training RandomForest (n_estimators=80, parallel)")
    rf  = RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=42, n_jobs=-1)
    clf = OneVsRestClassifier(rf, n_jobs=-1)
    clf.fit(Xtr, ytr)

    log.info("Evaluating")
    y_pred = clf.predict(Xte)
    print(classification_report(yte, y_pred, target_names=CATEGORIES))

    # 8) persist model + vectorizers
    joblib.dump({
        "model":    clf,
        "tfidf":    tfidf,
        "mlb":      mlb,
        "num_cols": num_cols
    }, MODEL_DIR/"threat_model_with_sbert.pkl")
    log.info(f"Saved model to {MODEL_DIR/'threat_model_with_sbert.pkl'}")

if __name__ == "__main__":
    main()
