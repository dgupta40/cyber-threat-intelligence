#!/usr/bin/env python3
"""
clean_text.py — Pre-process raw scraped data for modelling.

Pipeline:
1. Load flat JSON files from NVD + THN.
2. Clean & tokenize text (THN & NVD).
3. Extract CVEs (or use from THN directly).
4. Generate embeddings: TF-IDF, Word2Vec, SBERT.
5. Compute sentiment.
6. Add temporal features.
7. Save master_<timestamp>.parquet and .csv.
"""

from __future__ import annotations
import json, logging, re
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

# ── Paths & constants ────────────────────────────
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
for p in (PROC_DIR, MODEL_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "clean_text.log"), logging.StreamHandler()],
)

# ── Functions ──────────────────────────

def _clean_html(html) -> str:
    if not isinstance(html, str) or len(html.strip()) < 10:
        return ""
    try:
        return re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ", strip=True))
    except Exception:
        return ""

def _cybersecurity_normalize(text: str) -> str:
    text = re.sub(r'CVE-\d{4}-\d{4,7}', 'CVE_REFERENCE', text)
    text = re.sub(r'(\d+\.\d+\.\d+)', 'VERSION_NUMBER', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP_ADDRESS', text)
    patterns = {
        r'\b(sql injection|sqli)\b': 'SQL_INJECTION',
        r'\b(cross[- ]?site[- ]?scripting|xss)\b': 'XSS',
        r'\b(denial[- ]?of[- ]?service|dos|ddos)\b': 'DOS_ATTACK',
        r'\b(remote code execution|rce)\b': 'RCE',
        r'\b(local file inclusion|lfi)\b': 'LFI',
        r'\b(remote file inclusion|rfi)\b': 'RFI',
        r'\b(cross[- ]?site[- ]?request[- ]?forgery|csrf)\b': 'CSRF',
        r'\b(server[- ]?side[- ]?request[- ]?forgery|ssrf)\b': 'SSRF',
    }
    for pat, repl in patterns.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def _extract_cve_mentions(text: str) -> List[str]:
    return list(set(re.findall(r'CVE-\d{4}-\d{4,7}', text, flags=re.IGNORECASE)))

def _tokenise(txt: str) -> List[str]:
    txt = _cybersecurity_normalize(txt)
    tokens = [t.lower() for t in word_tokenize(txt) if t.isalpha()]
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS]

def _cvss_bin(score) -> str:
    try: s = float(score)
    except: return "unknown"
    if s < 4: return "low"
    if s < 7: return "medium"
    if s < 9: return "high"
    return "critical"

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['published_dt'] = pd.to_datetime(df['published'], errors='coerce')
    df['day_of_week'] = df['published_dt'].dt.dayofweek
    df['hour_of_day'] = df['published_dt'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['year'] = df['published_dt'].dt.year
    df['month'] = df['published_dt'].dt.month
    return df

def load_raw_newer_than(_: datetime) -> pd.DataFrame:
    records: List[Dict] = []
    for jf in RAW_DIR.rglob("*.json"):
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Skipping {jf} due to error: {e}")
            continue

        if isinstance(data, list) and data:
            src = data[0].get("source", "").lower().replace(" ", "")
            for rec in data:
                rec["source"] = "nvd" if src == "nvd" else "thehackernews"
                records.append(rec)

    return pd.DataFrame.from_records(records)

# ── Main ──────────────────────────

def _score(text: str, pipe) -> float:
    lab = pipe(text[:512])[0]
    if lab['label'].lower().startswith('pos'): return lab['score']
    if lab['label'].lower().startswith('neg'): return -lab['score']
    return 0.0

def main() -> str:
    last_ts = datetime(1970, 1, 1, tzinfo=timezone.utc)
    logging.info("Last processed: None (processing all files)")

    df = load_raw_newer_than(last_ts)
    if df.empty:
        logging.info("No raw data found - exiting.")
        return ""

    logging.info(f"Loaded {len(df):,} raw rows")
    df['source'] = df['source'].str.lower().str.replace(" ", "")
    df_thn = df[df['source'] == 'thehackernews'].copy()
    df_nvd = df[df['source'] == 'nvd'].copy()

    # NVD processing
    df_nvd['cvssScore'] = pd.to_numeric(df_nvd.get('cvss_score'), errors='coerce')
    df_nvd['severity_bin'] = df_nvd['cvssScore'].apply(_cvss_bin)
    df_nvd['description'] = df_nvd.get('description', '')
    df_nvd['clean_text'] = df_nvd['description'].fillna('').map(_clean_html)
    df_nvd['tokens'] = df_nvd['clean_text'].apply(_tokenise)
    df_nvd['sbert_path'] = ''
    df_nvd['sentiment'] = np.nan
    df_nvd['mentioned_cves'] = df_nvd['cve_id'].apply(lambda x: [x] if pd.notna(x) else [])
    df_nvd = _add_temporal_features(df_nvd.rename(columns={"published_date": "published"}))

    # THN processing
    df_thn['clean_text'] = df_thn['text'].fillna('').map(_clean_html)
    df_thn['mentioned_cves'] = df_thn['cves'].apply(lambda x: list(set([c.upper() for c in x])) if isinstance(x, list) else [])
    tqdm.pandas(desc="tokenising")
    df_thn['tokens'] = df_thn['clean_text'].progress_map(_tokenise)

    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df_thn['clean_text'])
    pd.to_pickle({'model': tfidf, 'matrix': X_tfidf}, MODEL_DIR / f'tfidf_thn_{datetime.utcnow():%Y%m%d_%H%M}.pkl')

    w2v = Word2Vec(df_thn['tokens'], vector_size=200, window=5, sg=1, min_count=3)
    w2v.save(str(MODEL_DIR / f'w2v_thn_{datetime.utcnow():%Y%m%d_%H%M}.model'))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    emb = sbert.encode(df_thn['clean_text'].tolist(), batch_size=64, device="cuda" if sbert.device.type == 'cuda' else 'cpu', show_progress_bar=True).astype('float32')
    path_s = MODEL_DIR / f'sbert_thn_{datetime.utcnow():%Y%m%d_%H%M}.npy'
    np.save(path_s, emb)
    df_thn['sbert_path'] = str(path_s)

    sent_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0 if sbert.device.type == 'cuda' else -1)
    tqdm.pandas(desc="sentiment")
    df_thn['sentiment'] = df_thn['clean_text'].progress_map(lambda text: _score(text, sent_pipe))
    df_thn = _add_temporal_features(df_thn.rename(columns={"published_date": "published"}))

    df_master = pd.concat([df_nvd, df_thn]).sort_index()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_parquet = PROC_DIR / f"master_{timestamp}.parquet"
    out_csv = PROC_DIR / f"master_{timestamp}.csv"
    df_master.to_parquet(out_parquet, index=False)
    df_master.to_csv(out_csv, index=False)
    logging.info(f"Saved master -> {out_parquet}\n CSV version saved -> {out_csv}")

    # export THN and NVD separately
    df_thn.to_csv(PROC_DIR / f"thn_cleaned_{timestamp}.csv", index=False)
    df_nvd.to_csv(PROC_DIR / f"nvd_cleaned_{timestamp}.csv", index=False)

    return str(out_parquet)

class TextPreprocessor:
    def run(self): return main()
    process_all_sources = run

if __name__ == '__main__':
    main()
