#!/usr/bin/env python3
"""
clean_text.py — Pre-process raw scraped data for modelling.

Pipeline
1. Load *new* raw JSON/JSONL (THN & NVD v2) with source tagging.
2. Extract NVD CVSS v3 scores → severity_bin (NVD only).
3. Clean HTML → tokens (THN only).
4. Build embeddings: TF-IDF, Word2Vec, SBERT (THN only).
5. Compute sentiment polarity (THN only).
6. Save combined master_<timestamp>.parquet.

Note: schemas.yml is not directly used here; loader uses code-based detection of feed schemas.
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
lemmatizer = WordNetLemmatizer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "clean_text.log"), logging.StreamHandler()],
)

# ── Helper functions ───────────────────────────────────────────────────────────
def _extract_nvd_score(row) -> float | None:
    metrics = row.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if isinstance(entries, list) and entries:
            cvss_data = entries[0].get("cvssData", {})
            try:
                return float(cvss_data.get("baseScore"))
            except (TypeError, ValueError, KeyError):
                continue
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

# ── Load new raw rows with source tagging ───────────────────────────────────────
def load_raw_newer_than(ts_last: datetime) -> pd.DataFrame:
    ts_epoch = ts_last.timestamp()
    records: List[Dict] = []

    # JSONL.GZ (if you ever get gzipped JSONL feeds)
    for gz in RAW_DIR.rglob("*.jsonl.gz"):
        if gz.stat().st_mtime <= ts_epoch:
            continue
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["source"] = rec.get("source","unknown")
                records.append(rec)

    # JSON files: could be THN or NVD v2.0 (REST API)
    for jf in RAW_DIR.rglob("*.json"):
        if jf.stat().st_mtime <= ts_epoch:
            continue
        data = json.load(open(jf, encoding="utf-8"))

        # 1) NVD v2.0 REST API format under 'vulnerabilities'
        if isinstance(data, dict) and "vulnerabilities" in data:
            for rec in data["vulnerabilities"]:
                rec["source"] = "nvd"
                records.append(rec)

        # 2) THN list of articles
        elif isinstance(data, list) and data and data[0].get("source")=="thehackernews":
            for rec in data:
                rec["source"] = "thehackernews"
                records.append(rec)

        # 3) Legacy NVD v1 schema with 'CVE_Items'
        elif isinstance(data, dict) and "CVE_Items" in data:
            for rec in data["CVE_Items"]:
                rec["source"] = "nvd"
                records.append(rec)

    return pd.DataFrame.from_records(records)

# ── Main pipeline ──────────────────────────────────────────────────────────────
def main() -> str:
    latest = max(PROC_DIR.glob("master_*.parquet"), default=None)
    last_ts = (
        datetime.utcfromtimestamp(latest.stat().st_mtime) if latest
        else datetime(1970,1,1,tzinfo=timezone.utc)
    )
    logging.info(f"Last processed: {latest}, ts={last_ts}")

    df = load_raw_newer_than(last_ts)
    if df.empty:
        logging.info("No new raw data - exiting.")
        return str(latest) if latest else ""
    logging.info(f"Loaded {len(df):,} raw rows")

    # Split by source
    df_thn = df[df['source']=='thehackernews'].copy()
    df_nvd = df[df['source']=='nvd'].copy()

    # NVD-only: extract CVSS and severity
    df_nvd['cvssScore']   = df_nvd.apply(_extract_nvd_score, axis=1)
    df_nvd['severity_bin'] = df_nvd['cvssScore'].apply(_cvss_bin)

    # THN-only: clean & tokenise text
    df_thn['content']    = df_thn.get('content','').fillna('')
    tqdm.pandas(desc='clean_html')
    df_thn['clean_text'] = df_thn['content'].progress_map(_clean_html)
    tqdm.pandas(desc='tokenise')
    df_thn['tokens']     = df_thn['clean_text'].progress_map(_tokenise)

    # THN-only embeddings & sentiment
    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df_thn['clean_text'])
    pd.to_pickle({'model':tfidf,'matrix':X_tfidf},
                 MODEL_DIR/f'tfidf_thn_{datetime.utcnow():%Y%m%d_%H%M}.pkl')

    w2v = Word2Vec(df_thn['tokens'], vector_size=200, window=5, sg=1, min_count=3)
    w2v.save(str(MODEL_DIR/f'w2v_thn_{datetime.utcnow():%Y%m%d_%H%M}.model'))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    emb = sbert.encode(
        df_thn['clean_text'].tolist(),
        batch_size=64,
        device="cuda" if sbert.device.type=='cuda' else 'cpu',
        show_progress_bar=True
    ).astype('float32')
    path_s = MODEL_DIR/f'sbert_thn_{datetime.utcnow():%Y%m%d_%H%M}.npy'
    np.save(path_s, emb)
    df_thn['sbert_path'] = str(path_s)

    # Sentiment pipeline
    sent_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0 if sbert.device.type=='cuda' else -1,
    )

    def _score(text: str) -> float:
        lab = sent_pipe(text[:512])[0]
        if lab['label'].lower().startswith('pos'):
            return lab['score']
        if lab['label'].lower().startswith('neg'):
            return -lab['score']
        return 0.0

    tqdm.pandas(desc='sentiment')
    df_thn['sentiment'] = df_thn['clean_text'].progress_map(_score)

    # Fill dummy columns for NVD so concat works
    df_nvd['content']    = ''
    df_nvd['clean_text'] = ''
    df_nvd['tokens']     = None
    df_nvd['sbert_path'] = ''
    df_nvd['sentiment']  = np.nan

    # Recombine and save master
    df_master = pd.concat([df_nvd, df_thn]).sort_index()
    out = PROC_DIR/f"master_{datetime.utcnow():%Y%m%d_%H%M}.parquet"
    df_master.to_parquet(out, index=False)
    logging.info(f"Saved master -> {out}")
    return str(out)


# Back‑compat wrapper
class TextPreprocessor:
    def run(self): return main()
    process_all_sources = run

if __name__ == '__main__':
    main()
