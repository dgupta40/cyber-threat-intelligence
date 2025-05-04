from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def _unique_name(source: str, obj: dict) -> str:
    """
    Build an OS-safe, collision-free filename for each record:
      • NVD  →  nvd_<CVE-ID>.json
      • others → <source>_<md5(url)>.json
    """
    if source == "nvd" and obj.get("cve", {}).get("id"):
        return f"{source}_{obj['cve']['id']}.json"

    url = obj.get("url", str(obj)[:50])
    digest = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{source}_{digest}.json"

def save_raw(obj: dict, source: str):
    """
    Save one record only if it hasn't been stored before.
    """
    fname = RAW_DIR / _unique_name(source, obj)
    if fname.exists():
        # already scraped in a previous run – skip
        return
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


# ---------- simple HTTP helper -------------------------------
import requests, time

def safe_request(url: str, timeout: int = 15, retries: int = 3) -> requests.Response:
    """
    GET with retry/back-off and a desktop UA string.
    Raises the last exception if all retries fail.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/124.0 Safari/537.36"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_err = exc
            time.sleep(1 + attempt)          # linear back-off
    raise last_err

