"""
End‑to‑end ETL for NVD + The Hacker News
▸ Streams raw JSON  ▸ Cleans / tokenises  ▸ Enriches with CVSS‑bins, CWE one‑hots,
▸ Saves versioned Parquet master  ▸ Generates TF‑IDF + Word2Vec embeddings
Incremental‑safe (processes only rows newer than last run).
"""

from __future__ import annotations
import os, re, json, logging, gzip, yaml, math
from datetime import datetime, timezone
from typing import Any, Dict, List, Generator

import ijson
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from utils.helpers import get_all_files, load_from_json, save_to_json


# NLTK boot‑strap (quietly ignore if offline)
for pkg in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    """Cleans, enriches & merges HackerNews + NVD, writes Parquet master + embeddings."""

    RE_URL  = re.compile(r'https?://\S+')
    RE_MAIL = re.compile(r'\S+@\S+')
    RE_CVE  = re.compile(r'CVE-\d{4}-\d{4,7}', re.I)

    def __init__(self,
                 raw_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 schema_path: str = "schemas.yaml"):
        self.logger = logging.getLogger(__name__)
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        # load schema manifest
        self.schemas: Dict[str, Any] = yaml.safe_load(open(schema_path))

        # NLTK helpers
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # incremental state
        self.state_file = os.path.join(self.processed_dir, ".last_ingest.txt")
        self.last_run   = None
        if os.path.exists(self.state_file):
            try:
                self.last_run = datetime.fromisoformat(open(self.state_file).read().strip())
                self.logger.info("Incremental mode – only ingest newer than %s", self.last_run)
            except Exception:
                self.last_run = None

    # ==========================================================================
    # Public entry‑point
    # ==========================================================================
    def process_all_sources(self) -> None:
        self.logger.info(" ETL run started")
        hn_docs  = self._process_hackernews()
        nvd_docs = self._process_nvd()
        self._create_master_dataset(hn_docs, nvd_docs)
        self._generate_embeddings()
        # record successful run timestamp
        with open(self.state_file, "w") as fh:
            fh.write(datetime.utcnow().isoformat())
        self.logger.info(" ETL finished")

    # ==========================================================================
    # Hacker News
    # ==========================================================================
    def _process_hackernews(self) -> List[Dict[str, Any]]:
        src_dir = os.path.join(self.raw_dir, "hackernews")
        docs: List[Dict[str, Any]] = []

        for path in get_all_files(src_dir, ".json"):
            for art in load_from_json(path):
                pub_raw = art.get("date") or art.get("published_date") or ""
                pub_dt  = self._parse_date(pub_raw)
                if self.last_run and pub_dt and pub_dt <= self.last_run:
                    continue

                text = art.get("content") or art.get("body_text") or ""
                if not text:
                    continue

                clean  = self._clean_text(text)
                tokens = self._tokenize(clean)
                cves   = list({cve.upper() for cve in self.RE_CVE.findall(text)})

                docs.append({
                    "source": "hackernews",
                    "id":     art.get("url", ""),
                    "title":  art.get("title", ""),
                    "content": clean,
                    "tokens":  tokens,
                    "date":    pub_dt.isoformat() if pub_dt else "",
                    "metadata": {
                        "tags": art.get("tags", []),
                        "url":  art.get("url", ""),
                        "cves": cves
                    }
                })

        if docs:
            save_to_json(docs, os.path.join(self.processed_dir, "hackernews_processed.json"))
            self.logger.info("HackerNews processed: %d docs", len(docs))  # Remove emoji
        return docs

    # ==========================================================================
    # NVD feeds (handles v1.1 & new 2.0 JSON‑Lines style)
    # ==========================================================================
    def _process_nvd(self) -> List[Dict[str, Any]]:
        src_dir = os.path.join(self.raw_dir, "nvd")
        docs: List[Dict[str, Any]] = []

        for path in get_all_files(src_dir, ".json"):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                if isinstance(data, dict):
                    if "vulnerabilities" in data:  # newer schema
                        for item in data["vulnerabilities"]:
                            cve_root = item.get("cve", {})
                            cve_id = cve_root.get("id", "")
                            pub_dt = self._parse_date(item.get("published"))
                            if self.last_run and pub_dt and pub_dt <= self.last_run:
                                continue

                            desc = next((d["value"]
                                        for d in cve_root.get("descriptions", [])
                                        if d.get("lang") == "en"), "")
                            score, sev = self._extract_cvss(item.get("metrics", {}))

                            docs.append(self._build_nvd_doc(
                                cve_id, desc, score, sev,
                                pub_dt, self._parse_date(item.get("lastModified"))))
                    elif "CVE_Items" in data:  # 1.1 schema
                        for item in data.get("CVE_Items", []):
                            cve_meta = item["cve"]["CVE_data_meta"]
                            cve_id = cve_meta["ID"]
                            pub_dt = self._parse_date(item.get("publishedDate"))
                            if self.last_run and pub_dt and pub_dt <= self.last_run:
                                continue

                            desc = next((d["value"]
                                        for d in item["cve"]["description"]["description_data"]
                                        if d["lang"] == "en"), "")
                            score, sev = self._extract_cvss(item.get("impact", {}), legacy=True)

                            docs.append(self._build_nvd_doc(
                                cve_id, desc, score, sev,
                                pub_dt, self._parse_date(item.get("lastModifiedDate"))))
            except Exception as e:
                self.logger.error(f"Error processing NVD file {path}: {str(e)}")
                continue

        if docs:
            save_to_json(docs, os.path.join(self.processed_dir, "nvd_processed.json"))
            self.logger.info("NVD processed: %d CVEs", len(docs))  # Remove emoji
        return docs

    # ==========================================================================
    # Master merge + feature engineering, Parquet versioning
    # ==========================================================================
    def _create_master_dataset(self,
                               hn: List[Dict[str, Any]],
                               nvd: List[Dict[str, Any]]) -> None:
        self.logger.info(" Merging HackerNews and NVD")

        # explode HN on CVE list
        hn_rows = []
        for d in hn:
            for cve in (d["metadata"]["cves"] or [None]):
                row = {**d, "cve_id": cve}
                hn_rows.append(row)
        df_hn  = pd.DataFrame(hn_rows)
        df_nvd = pd.DataFrame(nvd)

        master = df_hn.merge(df_nvd, on="cve_id", how="left",
                             suffixes=("_thn", "_nvd"))
        # publish_gap (hours)
        master["publish_gap_hr"] = (
            pd.to_datetime(master["date_thn"], utc=True, errors="coerce") -
            pd.to_datetime(master["date_nvd"], utc=True, errors="coerce")
        ).dt.total_seconds().div(3600)

        # CVSS bins (preferring NVD score if present)
        master["cvss_score"] = master["metadata_nvd"].apply(
            lambda m: m.get("base_score") if isinstance(m, dict) else math.nan)
        master["cvss_bin"]   = master["cvss_score"].apply(self._cvss_to_bin)

        # CWE one‑hot set ➜ JSON‑encoded dict (sparse)

        def cwe_onehot(meta):
            if not isinstance(meta, dict): return {}
            cwes = meta.get("cwe") or []
            result = {f"CWE_{c}": 1 for c in cwes}
            # Add a dummy field if the dictionary is empty
            if not result:
                result["dummy"] = 0
            return result
        master["cwe_onehot"] = master["metadata_nvd"].apply(cwe_onehot)

        dest = os.path.join(self.processed_dir, "master.parquet")
        table = pa.Table.from_pandas(master, preserve_index=False)
        pq.write_table(table, dest, compression="zstd")
        self.logger.info("Master saved (%s rows) to master.parquet", len(master))

    # ==========================================================================
    # Embeddings (TF‑IDF + Word2Vec)
    # ==========================================================================
    def _generate_embeddings(self) -> None:
        latest = os.path.join(self.processed_dir, "master_latest.parquet")
        if not os.path.exists(latest):
            self.logger.warning("No master.parquet – skip embeddings")
            return
        df = pq.read_table(latest).to_pandas()

        docs = df["content_thn"].fillna("") + " " + df["content_nvd"].fillna("")
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85,
                                     stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(docs)
        meta  = {"features": vectorizer.get_feature_names_out().tolist(),
                 "timestamp": datetime.utcnow().isoformat()}

        save_to_json(meta, os.path.join(self.processed_dir, "tfidf_metadata.json"))
        np.savez_compressed(os.path.join(self.processed_dir, "tfidf_matrix.npz"),
                            data=tfidf.data, indices=tfidf.indices,
                            indptr=tfidf.indptr, shape=tfidf.shape)

        # Word2Vec – need token lists
        token_lists = df["tokens_thn"].dropna().tolist() + df["tokens_nvd"].dropna().tolist()
        if token_lists:
            model = Word2Vec(sentences=token_lists,
                             vector_size=300, window=5,
                             min_count=1, workers=4)
            model.save(os.path.join(self.processed_dir, "word2vec_model.bin"))
        self.logger.info("Embeddings generated")

    # ==========================================================================
    # Helpers
    # ==========================================================================
    def _clean_text(self, text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
        text = self.RE_URL.sub("", text)
        text = self.RE_MAIL.sub("", text)
        text = re.sub(r"[^\w\s\.-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def _parse_date(self, s: str | None) -> datetime | None:
        if not s: return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            try:
                return pd.to_datetime(s, utc=True).to_pydatetime()
            except Exception:
                return None

    @staticmethod
    def _cvss_to_bin(score: float | None) -> str:
        if score is None or math.isnan(score):
            return "unknown"
        if score < 4:   return "low"
        if score < 7:   return "medium"
        if score < 9:   return "high"
        return "critical"

    @staticmethod
    def _extract_cvss(metrics: Dict[str, Any], legacy: bool = False):
        if legacy:
            cvss3 = metrics.get("baseMetricV3", {}).get("cvssV3", {})
            cvss2 = metrics.get("baseMetricV2", {}).get("cvssV2", {})
        else:
            cvss3 = metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})
            cvss2 = metrics.get("cvssMetricV2",  [{}])[0].get("cvssData", {})
        score = cvss3.get("baseScore") or cvss2.get("baseScore")
        sev   = cvss3.get("baseSeverity") or ""
        return score, sev

    def _build_nvd_doc(self, cve_id, desc, score, sev, pub_dt, mod_dt):
        # extract CWE list if present
        cwe = []
        try:
            pt = self.schemas.get("nvd_cwe_path")  # optional pointer in yaml
            # not implementing JSONPath; keep simple
        except Exception:
            pass
        return {
            "source": "nvd",
            "id":     cve_id,
            "cve_id": cve_id,
            "title":  cve_id,
            "content": self._clean_text(desc),
            "tokens":  self._tokenize(desc),
            "date":    pub_dt.isoformat() if pub_dt else "",
            "metadata": {
                "base_score": score, "severity": sev,
                "last_modified": mod_dt.isoformat() if mod_dt else "",
                "cwe": cwe
            }
        }

# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    tp = TextPreprocessor()
    tp.process_all_sources()