#!/usr/bin/env python3
"""
clean_text.py — CTI Data Preprocessing 

Comprehensive preprocessing pipeline for threat intelligence analysis:
- Multiple embedding techniques (TF-IDF, Word2Vec, SBERT)
- Domain-specific text normalization
- Quality validation
- Data integration features
- Automatic GPU detection
"""

from __future__ import annotations
import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import pipeline
import torch

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download failed: {e}")

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ── Paths & constants ─────────────────────────────────────────
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
    handlers=[logging.FileHandler(
        LOG_DIR / "clean_text.log"), logging.StreamHandler()],
)

# ── Text cleaning functions ────────────────────────────


def _clean_html(html) -> str:
    """Extract and clean text from HTML content"""
    if not isinstance(html, str) or len(html.strip()) < 10:
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception as e:
        logging.warning(f"HTML cleaning failed: {e}")
        return html


def _clean_text(text: str) -> str:
    """Comprehensive text cleaning for CTI data"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Remove hash values (MD5, SHA256)
    text = re.sub(r'\b[a-fA-F0-9]{32}\b', 'MD5_HASH', text)
    text = re.sub(r'\b[a-fA-F0-9]{64}\b', 'SHA256_HASH', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _cybersecurity_normalize(text: str) -> str:
    """Normalize cybersecurity terms and concepts"""
    # Normalize CVE references
    text = re.sub(r'CVE-\d{4}-\d{4,7}', 'CVE_REFERENCE', text)
    
    # Normalize version numbers
    text = re.sub(r'(\d+\.\d+\.\d+)', 'VERSION_NUMBER', text)
    
    # Normalize IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP_ADDRESS', text)
    
    # Normalize common attack types
    attack_patterns = {
        r'\b(sql injection|sqli)\b': 'SQL_INJECTION',
        r'\b(cross[- ]?site[- ]?scripting|xss)\b': 'XSS',
        r'\b(denial[- ]?of[- ]?service|dos|ddos)\b': 'DOS_ATTACK',
        r'\b(remote code execution|rce)\b': 'RCE',
        r'\b(cross[- ]?site[- ]?request[- ]?forgery|csrf)\b': 'CSRF',
    }
    
    for pattern, replacement in attack_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def _extract_cve_mentions(text: str) -> List[str]:
    """Extract CVE IDs from text"""
    return list(set(re.findall(r'CVE-\d{4}-\d{4,7}', text, flags=re.IGNORECASE)))


def _tokenise(txt: str) -> List[str]:
    """Tokenize and lemmatize text"""
    txt = _cybersecurity_normalize(txt)
    tokens = [t.lower() for t in word_tokenize(txt) if t.isalpha()]
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS]


def _cvss_bin(score) -> str:   ### Literature for this
    """Categorize CVSS scores into severity levels"""
    try:
        s = float(score)
        if s < 4:
            return "low"
        elif s < 7:
            return "medium"
        elif s < 9:
            return "high"
        else:
            return "critical"
    except:
        return "unknown"


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features for analysis"""
    df['published_dt'] = pd.to_datetime(df['published'], errors='coerce')
    df['year'] = df['published_dt'].dt.year
    df['month'] = df['published_dt'].dt.month
    df['day_of_week'] = df['published_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    return df


def _add_cyber_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cybersecurity-specific features"""
    df['has_exploit_mention'] = df['clean_text'].str.contains(
        r'exploit|poc|proof of concept', case=False, na=False)
    df['has_patch_mention'] = df['clean_text'].str.contains(
        r'patch|fix|update|remediat', case=False, na=False)
    df['attack_type_count'] = df['clean_text'].str.count(
        r'SQL_INJECTION|XSS|DOS_ATTACK|RCE|CSRF')
    return df


def _validate_cleaning(df: pd.DataFrame, source: str) -> None:
    """Validate text cleaning quality"""
    # Check for empty texts
    empty_count = df['clean_text'].str.strip().eq('').sum()
    if empty_count > 0:
        logging.warning(f"{source}: Found {empty_count} empty texts")
    
    # Check for very short texts
    short_count = df['clean_text'].str.len().lt(50).sum()
    if short_count > 0:
        logging.warning(f"{source}: Found {short_count} very short texts (<50 chars)")
    
    # Check for remaining HTML
    html_count = df['clean_text'].str.contains('<[^>]+>', regex=True).sum()
    if html_count > 0:
        logging.warning(f"{source}: Found {html_count} texts with potential HTML")


def load_raw_data() -> pd.DataFrame:
    """Load raw JSON data from NVD and TheHackerNews"""
    records = []
    
    for json_file in RAW_DIR.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for record in data:
                    source = record.get("source", "").lower()
                    if "nvd" in source:
                        record["source"] = "nvd"
                    else:
                        record["source"] = "thehackernews"
                    records.append(record)
        except Exception as e:
            logging.warning(f"Error loading {json_file}: {e}")
    
    return pd.DataFrame(records)

# ── Main processing pipeline ────────────────────────────


def batch_sentiment(texts, model, batch_size=32):
    """Process sentiment in batches for better performance"""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment batches"):
        batch = texts[i:i + batch_size]
        # Truncate to 512 tokens (BERT limit)
        batch_truncated = [text[:512] if text else "" for text in batch]
        try:
            batch_results = model(batch_truncated, truncation=True, padding=True)
            for result in batch_results:
                label_map = {'LABEL_0': -1.0, 'LABEL_1': 0.0, 'LABEL_2': 1.0}
                score = result['score'] * label_map.get(result['label'], 0.0)
                results.append(score)
        except Exception as e:
            logging.warning(f"Batch sentiment failed: {e}")
            results.extend([0.0] * len(batch))
    return results


def main() -> str:
    logging.info("Starting CTI data preprocessing...")
    
    # Load data
    df = load_raw_data()
    if df.empty:
        logging.error("No data found to process")
        return ""
    
    logging.info(f"Loaded {len(df)} records")
    
    # Split by source
    df_nvd = df[df['source'] == 'nvd'].copy()
    df_thn = df[df['source'] == 'thehackernews'].copy()
    
    # Detect GPU availability
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    
    # Initialize models
    sentiment_model = pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment", 
                              device=device)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info(f"SBERT using device: {sbert_model.device}")
    
    # Process NVD data
    logging.info(f"Processing {len(df_nvd)} NVD records...")
    df_nvd['cvss_score'] = pd.to_numeric(df_nvd.get('cvss_score', 0), errors='coerce')
    df_nvd['severity_category'] = df_nvd['cvss_score'].apply(_cvss_bin)
    
    # Clean text
    df_nvd['clean_text'] = df_nvd['description'].fillna('').apply(
        lambda x: _cybersecurity_normalize(_clean_text(_clean_html(x))))
    _validate_cleaning(df_nvd, "NVD")
    
    # Tokenize
    df_nvd['tokens'] = df_nvd['clean_text'].apply(_tokenise)
    
    # Extract CVEs
    df_nvd['mentioned_cves'] = df_nvd['cve_id'].apply(lambda x: [x] if pd.notna(x) else [])
    
    # Sentiment analysis
    logging.info("Computing sentiment for NVD...")
    df_nvd['sentiment'] = batch_sentiment(df_nvd['clean_text'].tolist(), sentiment_model)
    
    # Generate embeddings
    logging.info("Generating SBERT embeddings for NVD...")
    nvd_embeddings = sbert_model.encode(df_nvd['clean_text'].tolist(), 
                                       batch_size=32,
                                       show_progress_bar=True)
    np.save(MODEL_DIR / 'sbert_nvd.npy', nvd_embeddings)
    
    # Process THN data
    logging.info(f"Processing {len(df_thn)} THN records...")
    
    # Clean text
    df_thn['clean_text'] = df_thn['text'].fillna('').apply(
        lambda x: _cybersecurity_normalize(_clean_text(_clean_html(x))))
    _validate_cleaning(df_thn, "THN")
    
    # Tokenize
    df_thn['tokens'] = df_thn['clean_text'].apply(_tokenise)
    
    # Extract CVEs
    df_thn['mentioned_cves'] = df_thn['cves'].apply(
        lambda x: list(set(x)) if isinstance(x, list) else []
    )
    
    # Extract additional CVEs from text
    df_thn['cves_from_text'] = df_thn['clean_text'].apply(_extract_cve_mentions)
    df_thn['all_cves'] = df_thn.apply(
        lambda row: list(set(row['mentioned_cves'] + row['cves_from_text'])), axis=1
    )
    
    # TF-IDF for THN
    logging.info("Generating TF-IDF features...")
    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df_thn['clean_text'])
    pd.to_pickle({'model': tfidf, 'matrix': X_tfidf}, MODEL_DIR / 'tfidf_thn.pkl')
    
    # Word2Vec for THN
    logging.info("Training Word2Vec model...")
    w2v = Word2Vec(df_thn['tokens'], vector_size=100, window=5, min_count=3)
    w2v.save(str(MODEL_DIR / 'w2v_thn.model'))
    
    # SBERT embeddings for THN
    logging.info("Generating SBERT embeddings for THN...")
    thn_embeddings = sbert_model.encode(df_thn['clean_text'].tolist(), 
                                       batch_size=32,
                                       show_progress_bar=True)
    np.save(MODEL_DIR / 'sbert_thn.npy', thn_embeddings)
    
    # Sentiment analysis
    logging.info("Computing sentiment for THN...")
    df_thn['sentiment'] = batch_sentiment(df_thn['clean_text'].tolist(), sentiment_model)
    
    # Add features
    df_nvd = _add_temporal_features(df_nvd.rename(columns={"published_date": "published"}))
    df_nvd = _add_cyber_features(df_nvd)
    
    df_thn = _add_temporal_features(df_thn.rename(columns={"published_date": "published"}))
    df_thn = _add_cyber_features(df_thn)
    
    # Standardize columns
    df_nvd = df_nvd.rename(columns={'cve_id': 'primary_cve'})
    df_thn['primary_cve'] = None
    df_thn['cvss_score'] = None
    df_thn['severity_category'] = 'unknown'
    
    # Create CVE mapping
    logging.info("Creating CVE to article mapping...")
    cve_to_articles = {}
    for idx, row in df_thn.iterrows():
        for cve in row.get('all_cves', []):
            if cve not in cve_to_articles:
                cve_to_articles[cve] = []
            cve_to_articles[cve].append(idx)
    
    import pickle
    with open(MODEL_DIR / 'cve_article_mapping.pkl', 'wb') as f:
        pickle.dump(cve_to_articles, f)
    
    # Combine datasets (with sort=False to avoid warning)
    df_master = pd.concat([df_nvd, df_thn], ignore_index=True, sort=False)
    
    # Save results
    output_file = PROC_DIR / "master.parquet"
    df_master.to_parquet(output_file, index=False)
    df_master.to_csv(PROC_DIR / "master.csv", index=False)
    
    # Save individual datasets
    df_nvd.to_csv(PROC_DIR / "nvd_cleaned.csv", index=False)
    df_thn.to_csv(PROC_DIR / "thn_cleaned.csv", index=False)
    
    # Save metadata
    metadata = {
        'processed_date': datetime.now().isoformat(),
        'nvd_records': len(df_nvd),
        'thn_records': len(df_thn),
        'total_records': len(df_master),
        'models_used': {
            'sbert': 'all-MiniLM-L6-v2',
            'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment',
            'word2vec': 'custom-100d',
            'tfidf': 'sklearn'
        }
    }
    
    with open(PROC_DIR / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Processing complete! Saved to {output_file}")
    logging.info(f"Total records: {len(df_master)} (NVD: {len(df_nvd)}, THN: {len(df_thn)})")
    
    return str(output_file)


class TextPreprocessor:
    """Wrapper class for compatibility with run.py"""
    
    def process_all_sources(self):
        """Process all data sources"""
        return main()
    
    def run(self):
        """Alternative method name"""
        return main()


if __name__ == '__main__':
    main()