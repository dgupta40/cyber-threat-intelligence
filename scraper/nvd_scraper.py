import requests, os, time, datetime, logging, json
from utils.helpers import save_raw
from utils.advisory import enrich_with_wpscan
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NVD_API_KEY")
HEADERS = {"apiKey": API_KEY} if API_KEY else {}
API     = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def _merge_external_refs(vuln: dict) -> dict:
    """Follow reference URLs we know how to enrich (currently WP-Scan)."""
    for ref in vuln["cve"].get("references", []):
        if url := ref.get("url"):
            vuln = enrich_with_wpscan(url, vuln)
    return vuln

def pull_nvd(start_date: str, end_date: str):
    params = {
        "pubStartDate": start_date,
        "pubEndDate":   end_date,
        "resultsPerPage": 2000,
    }
    page = 0
    while True:
        params["startIndex"] = page * params["resultsPerPage"]
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        vulns = data.get("vulnerabilities") or []
        if not vulns:
            break

        for v in vulns:
            v = _merge_external_refs(v)
            save_raw(v, "nvd")

        if (page + 1) * params["resultsPerPage"] >= data["totalResults"]:
            break
        page += 1
        time.sleep(1)
