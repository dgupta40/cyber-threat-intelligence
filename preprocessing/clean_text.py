"""
clean_text.py — Pre-process raw scraped data for modelling.

Pipeline
1. Load *new* raw JSON/JSONL files.
2. Clean HTML -> tokens.
3. Pull CVSS scores (flat + nested) → severity_bin.
4. TF-IDF  ·  Word2Vec  ·  SBERT embeddings.
5. Sentiment polarity.
6. Save artefacts + master_<timestamp>.parquet.
"""

from __future__ import annotations
import gzip, json, logging, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import pipeline

# ── Paths & constants ──────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
LOG_DIR   = Path("logs")

for p in (PROC_DIR, MODEL_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

STOPWORDS = set(stopwords.words("english"))
CVSS_RE   = re.compile(r"(?:CVSS[:\s]*)(\d\.\d)", re.I)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "clean_text.log"), logging.StreamHandler()],
)

lemmatizer = WordNetLemmatizer()

# ── Helper functions ───────────────────────────────────────────────────────────
def _extract_nvd_score(row) -> float | None:
    impact = row.get("impact")
    if not isinstance(impact, dict):
        return None
    try:
        return float(
            impact.get("baseMetricV3", {})
                  .get("cvssV3", {})
                  .get("baseScore")
        )
    except (TypeError, ValueError):
        return None

def _clean_html(html) -> str:
    if not isinstance(html, str):
        return ""
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)

def _tokenise(txt: str) -> List[str]:
    tokens = [t.lower() for t in word_tokenize(txt) if t.isalpha()]
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS]

def _cvss_bin(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if s < 4:
        return "low"
    if s < 7:
        return "medium"
    if s < 9:
        return "high"
    return "critical"

# ── Load new raw rows since last parquet ───────────────────────────────────────
def load_raw_newer_than(ts_last: datetime) -> pd.DataFrame:
    ts_epoch = ts_last.timestamp()
    records: List[Dict] = []

    # jsonl.gz
    for gz in RAW_DIR.rglob("*.jsonl.gz"):
        if gz.stat().st_mtime <= ts_epoch:
            continue
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

    # bulk .json
    for jf in RAW_DIR.rglob("*.json"):
        if jf.stat().st_mtime <= ts_epoch:
            continue
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif "CVE_Items" in data:
                records.extend(data["CVE_Items"])

    return pd.DataFrame.from_records(records)

# ── Main pipeline ──────────────────────────────────────────────────────────────
def main() -> str:
    latest = max(PROC_DIR.glob("master_*.parquet"), default=None)
    last_ts = (
        datetime.utcfromtimestamp(latest.stat().st_mtime) if latest
        else datetime(1970, 1, 1, tzinfo=timezone.utc)
    )
    logging.info(f"Last processed: {latest}, ts={last_ts}")

    df = load_raw_newer_than(last_ts)
    if df.empty:
        logging.info("No new raw data - exiting.")
        return str(latest) if latest else ""

    logging.info(f"Loaded {len(df):,} raw rows")

    # ── Add/merge cvssScore from nested NVD fields ───────────────────
    if "cvssScore" not in df.columns:
        df["cvssScore"] = np.nan
    mask = df["cvssScore"].isna() & df["impact"].notna()
    if mask.any():
        df.loc[mask, "cvssScore"] = df.loc[mask].apply(_extract_nvd_score, axis=1)

    # ── Cleaning & tokenising ─────────────────────────────────────────
    df["content"] = df["content"].fillna("")
    tqdm.pandas(desc="clean_html")
    df["clean_text"] = df["content"].progress_map(_clean_html)

    tqdm.pandas(desc="tokenise")
    df["tokens"] = df["clean_text"].progress_map(_tokenise)

    # ── Severity bin ──────────────────────────────────────────────────
    df["severity_bin"] = df["cvssScore"].apply(_cvss_bin)

    # ── CWE one‑hot ───────────────────────────────────────────────────
    if "cweIds" in df.columns:
        df = pd.concat(
            [
                df.drop("cweIds", axis=1),
                (df["cweIds"].explode()
                               .astype("category")
                               .str.get_dummies(prefix="cwe")
                               .groupby(level=0)
                               .max()),
            ],
            axis=1,
        )

    # ── TF‑IDF ────────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df["clean_text"])
    tfidf_path = MODEL_DIR / f"tfidf_{datetime.utcnow():%Y%m%d_%H%M}.pkl"
    pd.to_pickle({"model": tfidf, "matrix": X_tfidf}, tfidf_path)
    logging.info(f"Saved TF-IDF -> {tfidf_path}")

    # ── Word2Vec ──────────────────────────────────────────────────────
    w2v = Word2Vec(df["tokens"], vector_size=200, window=5, sg=1, min_count=3)
    w2v_path = MODEL_DIR / f"w2v_{datetime.utcnow():%Y%m%d_%H%M}.model"
    w2v.save(str(w2v_path))
    logging.info(f"Saved Word2Vec -> {w2v_path}")

    # ── SBERT embeddings ─────────────────────────────────────────────
    logging.info("Encoding SBERT …")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    emb   = sbert.encode(
        df["clean_text"].tolist(),
        batch_size=64,
        device="cuda" if sbert.device.type == "cuda" else "cpu",
        show_progress_bar=True,
    ).astype("float32")
    sbert_path = MODEL_DIR / f"sbert_{datetime.utcnow():%Y%m%d_%H%M}.npy"
    np.save(sbert_path, emb)
    df["sbert_path"] = str(sbert_path)

    # ── Sentiment ────────────────────────────────────────────────────
    logging.info("Scoring sentiment …")
    sent_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0 if sbert.device.type == "cuda" else -1,
    )

    def _score(text: str) -> float:
        lab = sent_pipe(text[:512])[0]
        if lab["label"].lower().startswith("pos"):
            return  lab["score"]
        if lab["label"].lower().startswith("neg"):
            return -lab["score"]
        return 0.0

    tqdm.pandas(desc="sentiment")
    df["sentiment"] = df["clean_text"].progress_map(_score)

    # ── Persist master parquet ───────────────────────────────────────
    out = PROC_DIR / f"master_{datetime.utcnow():%Y%m%d_%H%M}.parquet"
    df.to_parquet(out, index=False)
    logging.info(f"Saved master → {out}")

    return str(out)

# Back‑compat wrapper for run.py
class TextPreprocessor:
    def process_all_sources(self): return main()
    run = process_all_sources

if __name__ == "__main__":
    main()
