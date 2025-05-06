"""
helpers.py
────────────────────────────────────────────────────────────────────────────
* Env / logging bootstrap
* File‑system helpers  (recursive get_all_files, streaming JSON loader, Parquet IO)
* Simple HTTP “safe_request” with retries + exponential back‑off
* Data‑frame utilities: merge, chunk, timestamp, text normalise
* Compatible with the new text_preprocessor.py (ijson streaming, gzip‑aware load)
"""

from __future__ import annotations
import os, sys, gzip, json, csv, logging, time, math, re
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

import pandas as pd
import numpy as np
import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import ijson                           # streaming JSON parser

# ──────────────────────────────────────────────────────────────────────────
#  1.  Environment & Logging
# ──────────────────────────────────────────────────────────────────────────
def load_env(dotenv_path: str | None = None) -> None:
    """
    Load .env variables and set sensible defaults (e.g. TF log level).
    Call this once, near program start‑up.
    """
    load_dotenv(dotenv_path)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def setup_logging(log_dir: str = "logs",
                  level: int = logging.INFO,
                  name: str = "cyber_threat_intel") -> None:
    """
    Configure root logging with rotating file + console handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    )

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [
        logging.FileHandler(log_file, encoding="utf‑8"),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logging.getLogger(__name__).info("Logging initialised → %s", log_file)


# ──────────────────────────────────────────────────────────────────────────
#  2.  File‑system helpers
# ──────────────────────────────────────────────────────────────────────────
def setup_directories(directories: List[str]) -> None:
    for d in directories:
        os.makedirs(d, exist_ok=True)


def get_all_files(root: str,
                  suffix: str | None = None,
                  recursive: bool = True) -> List[str]:
    """
    Recursively gather file‑paths, optionally filtered by suffix (e.g. '.json').
    """
    paths = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if suffix is None or fn.endswith(suffix):
                    paths.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            path = os.path.join(root, fn)
            if os.path.isfile(path) and (suffix is None or fn.endswith(suffix)):
                paths.append(path)
    return paths


# ── JSON helpers ─────────────────────────────────────────────────────────
def load_from_json(path: str) -> Any:
    """
    Gzip-aware JSON load. Falls back to streaming for >200 MB files.
    """
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    try:
        # heuristic: stream huge files instead of full read
        if os.path.getsize(path) > 200 * 1024 * 1024:
            with opener(path, "rb") as fh:
                # Fully consume the generator to ensure the file is properly read and closed
                return list(ijson.items(fh, ''))
        with opener(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning(f"Error loading JSON file {path}: {str(e)}")
        return {}


def save_to_json(obj: Any, path: str, **json_kwargs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf‑8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, **json_kwargs)


#  CSV helpers (unchanged, but fixed writer modes)
def save_to_csv(data, filename, headers: List[str] | None = None) -> None:
    mode = 'w'
    file_exists = os.path.isfile(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode, newline='', encoding='utf‑8') as f:
        if isinstance(data[0], dict):
            if headers is None:
                headers = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(f)
            if headers and not file_exists:
                writer.writerow(headers)
            writer.writerows(data)


def load_from_csv(filename: str, as_dict: bool = True):
    try:
        if as_dict:
            return pd.read_csv(filename).to_dict('records')
        return pd.read_csv(filename).values.tolist()
    except FileNotFoundError:
        logging.error("CSV not found: %s", filename)
        return []
    except Exception as exc:
        logging.error("CSV load error %s – %s", filename, exc)
        return []


# ── Parquet helpers ──────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str,
                 compression: str = "zstd",
                 overwrite: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if overwrite and os.path.exists(path):
        os.remove(path)
    pq.write_table(table, path, compression=compression)


def load_parquet(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


# ──────────────────────────────────────────────────────────────────────────
#  3.  Networking helper – robust GET with retry/back‑off
# ──────────────────────────────────────────────────────────────────────────
def safe_request(url: str,
                 max_retries: int = 3,
                 timeout: int = 10,
                 backoff_base: int = 2,
                 **kwargs):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except RequestException as e:
            logging.warning("Request %s failed (%d/%d): %s",
                            url, attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
            else:
                logging.error("Request giving up: %s", url)
                return None


# ──────────────────────────────────────────────────────────────────────────
#  4.  Misc data helpers
# ──────────────────────────────────────────────────────────────────────────
def merge_dataframes(dfs: List[pd.DataFrame],
                     on: str | List[str] | None = None) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    df = dfs[0]
    for other in dfs[1:]:
        df = (pd.merge(df, other, on=on, how='outer')
              if on else pd.concat([df, other], ignore_index=True))
    return df


def format_timestamp(ts: str | datetime | None = None,
                     fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            try:
                ts = datetime.strptime(ts, fmt)
            except ValueError:
                return ts
    return ts.strftime(fmt)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def get_env_variable(key: str, default: Any = None) -> Any:
    return os.getenv(key, default)


# ──────────────────────────────────────────────────────────────────────────
#  5.  Streaming NVD reader (used by text_preprocessor)
# ──────────────────────────────────────────────────────────────────────────
def stream_nvd_items(path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy‑yield CVE items from either NVD 1.1 or 2.0 JSON blobs.
    Works with .json or .json.gz
    """
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    with opener(path, "rb") as fh:
        if b'"CVE_Items"' in fh.peek(1000):
            # 1.1 schema – stream the array
            for item in ijson.items(fh, "CVE_Items.item"):
                yield item
        else:
            # 2.0 schema – root{"vulnerabilities":[{…},…]}
            for item in ijson.items(fh, "vulnerabilities.item"):
                yield item


# ──────────────────────────────────────────────────────────────────────────
#  6.  Public module init
# ──────────────────────────────────────────────────────────────────────────
__all__ = [
    # env / logging
    "load_env", "setup_logging",
    # fs helpers
    "setup_directories", "get_all_files",
    "load_from_json", "save_to_json",
    "save_to_csv", "load_from_csv",
    "save_parquet", "load_parquet",
    # net
    "safe_request",
    # misc
    "merge_dataframes", "format_timestamp", "normalize_text", "chunk_list",
    "get_env_variable",
    # nvd stream
    "stream_nvd_items"
]
