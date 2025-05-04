"""
Pull extra metadata from first-party advisory pages (currently: WP-Scan).

export:
    enrich_with_wpscan(ref_url: str, base_doc: dict) -> dict
"""

from __future__ import annotations
import requests, datetime, logging, re

WP_EXPORT_RE = re.compile(r"https://wpscan\.com/vulnerability/([0-9a-f\-]+)/?")

def enrich_with_wpscan(ref_url: str, base_doc: dict) -> dict:
    """
    If ref_url points to a WP-Scan advisory, download its JSON export and
    merge useful fields into base_doc. Otherwise returns base_doc unchanged.
    """
    m = WP_EXPORT_RE.match(ref_url)
    if not m:
        return base_doc

    advisory_id = m.group(1)
    export_url   = f"https://wpscan.com/vulnerability/{advisory_id}.json"
    try:
        r = requests.get(export_url, timeout=15)
        r.raise_for_status()
        js = r.json()

        base_doc["wp_scan"] = {
            "title":             js.get("title"),
            "type":              js.get("classification", {}).get("type"),
            "owasp":             js.get("classification", {}).get("owasp_top_10"),
            "cwe":               js.get("classification", {}).get("cwe"),
            "cvss":              js.get("cvss"),
            "fixed_in":          js.get("fixed_in"),
            "poc":               js.get("proof_of_concept"),
            "published":         js.get("timeline", {}).get("publicly_published"),
            "wpscan_url":        ref_url,
        }
        # If NVD description is short, append WP-Scan description
        if desc := js.get("description"):
            base_doc.setdefault("cve", {}).setdefault("descriptions", []).append(
                {"lang": "en", "value": f"(WP-Scan) {desc}"}
            )
    except Exception as exc:
        logging.warning("WP-Scan fetch failed for %s – %s", ref_url, exc)

    return base_doc
