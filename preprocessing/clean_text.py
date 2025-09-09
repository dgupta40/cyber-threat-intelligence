#!/usr/bin/env python3
"""
CTI preprocessing & THN→NVD linking
– HTML → text
– Cyber normalization → generic cleaning
– Stop‑word removal & lemmatization
– CVE reference extraction
– CVSS binning
– Sentiment (batch, cardiffnlp/twitter-roberta-base-sentiment)
– TF‑IDF on THN (min_df=3, max_df=0.8)
– SBERT embeddings (all-MiniLM-L6-v2)
– Integrated CVE->article linking (n_articles, linked_articles, earliest_article_date)
Outputs: data/processed/cti.db table 'master'
"""

import json
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import pipeline
from database import save_table

# CONFIGURATION

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# REGEX
URL_RE = re.compile(r"http[s]?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)

ATTACK_MAP = {
    r"\b(sql injection|sqli)\b": "SQL_INJECTION",
    r"\b(cross[- ]?site[- ]?scripting|xss)\b": "XSS",
    r"\b(denial[- ]?of[- ]?service|dos|ddos)\b": "DOS_ATTACK",
    r"\b(remote code execution|rce)\b": "RCE",
    r"\b(cross[- ]?site[- ]?request[- ]?forgery|csrf)\b": "CSRF",
}

# HELPERS


def clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def generic_clean(txt: str) -> str:
    """Mask URLs, emails, hashes and collapse whitespace."""
    txt = URL_RE.sub("URL", txt)
    txt = EMAIL_RE.sub("EMAIL", txt)
    txt = MD5_RE.sub("MD5_HASH", txt)
    txt = SHA256_RE.sub("SHA256_HASH", txt)
    return re.sub(r"\s+", " ", txt).strip()


def cyber_normalise(txt: str) -> str:
    """Replace CVE refs, IPs, versions, common attack terms."""
    txt = CVE_RE.sub("CVE_REFERENCE", txt)
    txt = re.sub(r"(\d+\.\d+\.\d+)", "VERSION_NUMBER", txt)
    txt = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP_ADDRESS", txt)
    for pat, repl in ATTACK_MAP.items():
        txt = re.sub(pat, repl, txt, flags=re.I)
    return txt


def cvss_bin(score) -> str:
    """Bin CVSS score into low/medium/high/critical."""  # NIST CVSS v3.0
    if score is None:
        return "unknown"
    try:
        s = float(score)
        if s < 4:
            return "low"
        if s < 7:
            return "medium"
        if s < 9:
            return "high"
        return "critical"
    except:
        return "unknown"


def batch_sent(texts: List[str], model, bs: int = 32) -> List[float]:
    """Run sentiment in batches; map LABEL_0→-1, LABEL_1→0, LABEL_2→+1."""
    label_map = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="sentiment"):
        batch = [t[:512] for t in texts[i : i + bs]]
        try:
            res = model(batch, truncation=True, padding=True)
            out.extend([r["score"] * label_map[r["label"]] for r in res])
        except Exception:
            out.extend([0.0] * len(batch))
    return out


def load_raw() -> pd.DataFrame:
    """Load all JSONs under data/raw into one DataFrame; normalize source."""
    recs = []
    for p in RAW_DIR.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in data if isinstance(data, list) else [data]:
                src = r.get("source", "").lower()
                r["source"] = "nvd" if "nvd" in src else "thehackernews"
                recs.append(r)
        except Exception:
            continue
    return pd.DataFrame(recs)


def remove_stopwords_and_lemmatize(txt: str) -> str:
    """
    Tokenize text, drop any stop-word, lemmatize each token,
    and re-join into a cleaned string.
    """
    tokens = word_tokenize(txt)
    processed = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok.lower() not in STOPWORDS
    ]
    return " ".join(processed)


# MAIN


def main():
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    df = load_raw()
    if df.empty:
        logging.error("No raw data found; aborting.")
        return

    # Split sources
    df_nvd = df[df.source == "nvd"].copy()
    df_thn = df[df.source == "thehackernews"].copy()

    # — NVD processing —
    df_nvd["cvss_score"] = pd.to_numeric(df_nvd.get("cvss_score", 0), errors="coerce")
    df_nvd["severity_category"] = df_nvd["cvss_score"].apply(cvss_bin)
    df_nvd["clean_text"] = (
        df_nvd["description"]
        .fillna("")
        .apply(clean_html)
        .apply(generic_clean)
        .apply(cyber_normalise)
        .apply(remove_stopwords_and_lemmatize)
    )
    # drop empty clean_text rows
    df_nvd = df_nvd[df_nvd.clean_text.str.strip().ne("")].reset_index(drop=True)
    df_nvd["mentioned_cves"] = df_nvd["cve_id"].apply(
        lambda x: [x] if pd.notna(x) else []
    )

    # — THN processing —
    df_thn["clean_text"] = (
        df_thn["text"]
        .fillna("")
        .apply(clean_html)
        .apply(generic_clean)
        .apply(cyber_normalise)
        .apply(remove_stopwords_and_lemmatize)
    )
    # drop empty clean_text rows
    df_thn = df_thn[df_thn.clean_text.str.strip().ne("")].reset_index(drop=True)
    df_thn["mentioned_cves"] = df_thn["cves"].apply(
        lambda x: list(set(x)) if isinstance(x, list) else []
    )
    df_thn["cves_from_text"] = df_thn["clean_text"].apply(
        lambda t: list(set(CVE_RE.findall(t)))
    )
    df_thn["all_cves"] = df_thn.apply(
        lambda r: list(set(r.mentioned_cves + r.cves_from_text)), axis=1
    )

    # — Sentiment —
    device = 0 if torch.cuda.is_available() else -1
    sent_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device,
        batch_size=32,
    )
    df_nvd["sentiment"] = batch_sent(df_nvd.clean_text.tolist(), sent_model)
    df_thn["sentiment"] = batch_sent(df_thn.clean_text.tolist(), sent_model)

    # — TF-IDF (THN only) —
    tfidf = TfidfVectorizer(min_df=3, max_df=0.8)
    X_tfidf = tfidf.fit_transform(df_thn.clean_text)
    pd.to_pickle({"model": tfidf, "matrix": X_tfidf}, MODEL_DIR / "tfidf_thn.pkl")

    # — SBERT embeddings —
    logging.warning("Computing SBERT embeddings")
    sbert = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(MODEL_DIR))
    emb_nvd = sbert.encode(
        df_nvd.clean_text.tolist(), batch_size=32, show_progress_bar=True
    )
    np.save(MODEL_DIR / "sbert_nvd.npy", emb_nvd)
    emb_thn = sbert.encode(
        df_thn.clean_text.tolist(), batch_size=32, show_progress_bar=True
    )
    np.save(MODEL_DIR / "sbert_thn.npy", emb_thn)

    # — CVE→THN linking —
    cve_map = {}
    for idx, row in df_thn.iterrows():
        for cve in row.all_cves:
            cve_map.setdefault(cve, []).append(idx)

    df_nvd["n_articles"] = df_nvd.cve_id.map(lambda c: len(cve_map.get(c, [])))
    df_nvd["linked_articles"] = df_nvd.cve_id.map(lambda c: cve_map.get(c, []))
    df_nvd["earliest_article_date"] = df_nvd.cve_id.map(
        lambda c: (
            df_thn.loc[cve_map[c], "published_date"].min() if c in cve_map else pd.NaT
        )
    )

    # — Combine & save —
    df_master = pd.concat([df_nvd, df_thn], ignore_index=True, sort=False)
    save_table(df_master, "master")
    logging.warning(f"Saved master table with {len(df_master)} rows.")


if __name__ == "__main__":
    main()
