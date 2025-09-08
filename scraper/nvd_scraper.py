import os
import logging
import json
import time
from datetime import datetime, timedelta
import requests

from utils.helpers import save_to_json, load_from_json


def _chunk_ranges(start: datetime, end: datetime, max_days: int = 120):
    """
    Split date range into chunks not exceeding max_days.

    Args:
        start: Start datetime
        end: End datetime
        max_days: Maximum days per chunk (NVD API limit is 120)

    Yields:
        Tuple of (chunk_start, chunk_end) datetime pairs
    """
    delta = timedelta(days=max_days)
    chunk_start = start
    while chunk_start < end:
        # Subtract 1 millisecond so intervals butt up against each other
        chunk_end = min(end, chunk_start + delta - timedelta(milliseconds=1))
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(milliseconds=1)


def load_cwe_database():
    """
    Load CWE database from JSON file.

    Returns:
        Dictionary mapping CWE IDs to descriptions
    """
    try:
        with open("cwe_database.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("CWE database not found. Using basic CWE information only.")
        # Fallback to common CWEs if database file is missing
        return {
            "CWE-20": "Improper Input Validation",
            "CWE-22": "Path Traversal",
            "CWE-79": "Cross-site Scripting",
            "CWE-89": "SQL Injection",
            "NVD-CWE-Other": "Weakness not included in CWE list",
            "NVD-CWE-noinfo": "Insufficient information to assign CWE",
        }


# Load CWE database once at module import
CWE_DATABASE = load_cwe_database()


def extract_cve_record(v: dict) -> dict:
    """
    Extract and transform a raw NVD vulnerability entry into a clean record.

    Args:
        v: Raw vulnerability data from NVD API

    Returns:
        Dictionary with cleaned CVE data in standardized format
    """
    # Extract core CVE data
    c = v.get("cve", {})

    # Get metrics (located inside the 'cve' object)
    metrics = c.get("metrics", {})

    # Initialize default values
    cvss_score = None
    severity = "Unknown"

    # Check all CVSS versions in order of preference (newest first)
    cvss_versions = [
        "cvssMetricV40",  # CVSS 4.0
        "cvssMetricV31",  # CVSS 3.1
        "cvssMetricV30",  # CVSS 3.0
        "cvssMetricV2",  # CVSS 2.0
    ]

    # Extract CVSS score and severity from the first available version
    for metric_key in cvss_versions:
        if metric_key in metrics:
            metric_list = metrics[metric_key]
            if isinstance(metric_list, list) and len(metric_list) > 0:
                metric_data = metric_list[0]  # Use primary/NVD metric

                if "cvssData" in metric_data:
                    cvss_data = metric_data["cvssData"]
                    cvss_score = cvss_data.get("baseScore")

                    # Severity location varies by CVSS version
                    if metric_key in [
                        "cvssMetricV30",
                        "cvssMetricV31",
                        "cvssMetricV40",
                    ]:
                        severity = cvss_data.get("baseSeverity", "Unknown")
                    else:  # CVSS v2
                        severity = metric_data.get("baseSeverity", "Unknown")

                    if cvss_score is not None:
                        break

    # Extract affected products from CPE configurations
    products = []
    for config in c.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("vulnerable", False):
                    cpe_parts = cpe_match.get("criteria", "").split(":")
                    if len(cpe_parts) > 5:
                        vendor = cpe_parts[3]
                        product = cpe_parts[4]
                        version = cpe_parts[5]

                        # Format product name for readability
                        product_name = product.replace("_", " ").title()

                        # Add version information if specific
                        if version != "*" and version != "-":
                            if "versionEndIncluding" in cpe_match:
                                version_end = cpe_match["versionEndIncluding"]
                                full_product = f"{product_name} (up to {version_end})"
                            else:
                                full_product = f"{product_name} {version}"
                        else:
                            full_product = product_name

                        # Include vendor name if not redundant
                        if vendor not in product.lower() and vendor != "*":
                            vendor_name = vendor.replace("_", " ").title()
                            full_product = f"{vendor_name} {full_product}"

                        products.append(full_product)

    # Extract English description
    description = ""
    for desc in c.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    # Extract CWE with human-readable description
    cwe = "No CWE Identified"
    if c.get("weaknesses") and len(c.get("weaknesses", [])) > 0:
        weakness_entries = c["weaknesses"][0].get("description", [])
        if weakness_entries and len(weakness_entries) > 0:
            cwe_id = weakness_entries[0].get("value", "Unknown")

            # Add description if available in  database
            if cwe_id in CWE_DATABASE:
                cwe = f"{cwe_id}: {CWE_DATABASE[cwe_id]}"
            else:
                cwe = cwe_id

    # Extract dates
    published = c.get("published", "")
    last_mod = c.get("lastModified", "")

    # Return cleaned record
    return {
        "source": "NVD",
        "cve_id": c.get("id"),
        "description": description,
        "published_date": published[:10] if published else "",
        "last_modified": last_mod[:10] if last_mod else "",
        "severity": severity,
        "cvss_score": cvss_score,
        "cwe": cwe,
        "products": list(set(products)),  # Remove duplicates
    }


class NVDScraper:
    """
    Scraper for fetching CVE data from the NVD API 2.0.

    Handles both full dataset downloads and incremental updates,
    respecting API rate limits and date range restrictions.
    """

    def __init__(
        self,
        start_year: int = 2019,
        history_file: str = "data/raw/nvd/nvd_cleaned.json",
    ):
        """
        Initialize the NVD scraper.

        Args:
            start_year: First year to fetch CVEs from
            history_file: Path to store the cleaned CVE data
        """
        self.output_dir = os.path.dirname(history_file)
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.api_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.history_path = history_file
        self.start_year = start_year

        # Configure API key and rate limiting
        self.api_key = os.getenv("NVD_API_KEY")
        self.headers = {}

        if self.api_key:
            self.headers["apiKey"] = self.api_key
            self.rate_limit_delay = 0.6  # 50 requests per 30 seconds
        else:
            self.rate_limit_delay = 6.0  # 5 requests per 30 seconds
            self.logger.warning("No NVD_API_KEY found. Rate limits will be lower.")

    def fetch_all_cves(self) -> bool:
        """
        Fetch all CVEs from start_year to present.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(
            f"Fetching all CVEs from {self.start_year} to today via API 2.0"
        )
        raw_dict = {}
        current_year = datetime.utcnow().year

        # Fetch data year by year
        for year in range(self.start_year, current_year + 1):
            year_start = datetime(year, 1, 1)
            year_end = (
                datetime.utcnow()
                if year == current_year
                else datetime(year, 12, 31, 23, 59, 59, 999000)
            )

            # Split year into 120-day chunks (API limit)
            for chunk_start, chunk_end in _chunk_ranges(year_start, year_end):
                s = chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000")
                e = chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000")

                params = {"pubStartDate": s, "pubEndDate": e, "resultsPerPage": 2000}

                self.logger.info(f"Fetching CVEs for {s} -> {e}")

                # Fetch initial batch
                response = requests.get(
                    self.api_url, headers=self.headers, params=params
                )
                response.raise_for_status()
                data = response.json()
                total = data.get("totalResults", 0)

                if total == 0:
                    continue

                # Process first batch
                for v in data.get("vulnerabilities", []):
                    cid = v.get("cve", {}).get("id")
                    if cid:
                        raw_dict[cid] = v

                # Fetch remaining pages if necessary
                idx = len(data.get("vulnerabilities", []))
                while idx < total:
                    params["startIndex"] = idx
                    batch_resp = requests.get(
                        self.api_url, headers=self.headers, params=params
                    )
                    batch_resp.raise_for_status()
                    batch = batch_resp.json()

                    for v in batch.get("vulnerabilities", []):
                        cid = v.get("cve", {}).get("id")
                        if cid:
                            raw_dict[cid] = v

                    idx += len(batch.get("vulnerabilities", []))
                    time.sleep(self.rate_limit_delay)

            self.logger.info(
                f"Year {year} complete: Total CVEs so far: {len(raw_dict)}"
            )

        if not raw_dict:
            self.logger.error("No CVEs were found or processed!")
            return False

        # Transform and save cleaned records
        cleaned = [extract_cve_record(v) for v in raw_dict.values()]
        save_to_json(cleaned, self.history_path)
        self.logger.info(
            f"Fetch complete: {len(cleaned)} CVE records -> {self.history_path}"
        )
        return True

    def incremental_update(self) -> bool:
        """
        Update existing dataset with new and modified CVEs.

        Returns:
            bool: True if successful, False otherwise
        """
        # Load existing data
        try:
            existing = load_from_json(self.history_path)
            if not isinstance(existing, list):
                self.logger.warning("Existing history is not a list; resetting.")
                existing = []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading history file: {e}")
            self.logger.info("Starting fresh fetch")
            return self.fetch_all_cves()

        # Map existing records by CVE ID
        existing_map = {rec["cve_id"]: rec for rec in existing}

        # Find latest dates in existing data
        latest_pub = max(
            (datetime.fromisoformat(rec["published_date"]) for rec in existing),
            default=datetime(2000, 1, 1),
        )
        latest_mod = max(
            (datetime.fromisoformat(rec["last_modified"]) for rec in existing),
            default=datetime(2000, 1, 1),
        )

        new_count = 0
        mod_count = 0
        now = datetime.utcnow()

        # Helper function to process API responses
        def process_batch(batch):
            nonlocal new_count, mod_count
            for v in batch.get("vulnerabilities", []):
                cid = v.get("cve", {}).get("id")
                if not cid:
                    continue

                record = extract_cve_record(v)
                rec_pub = datetime.fromisoformat(record["published_date"])
                rec_mod = datetime.fromisoformat(record["last_modified"])

                if cid not in existing_map:
                    # New CVE
                    existing_map[cid] = record
                    new_count += 1
                elif rec_mod > datetime.fromisoformat(
                    existing_map[cid]["last_modified"]
                ):
                    # Updated CVE
                    existing_map[cid] = record
                    mod_count += 1

        # Fetch newly published CVEs
        start_pub = latest_pub + timedelta(seconds=1)
        for cstart, cend in _chunk_ranges(start_pub, now):
            s = cstart.strftime("%Y-%m-%dT%H:%M:%S.000")
            e = cend.strftime("%Y-%m-%dT%H:%M:%S.000")
            params = {"pubStartDate": s, "pubEndDate": e, "resultsPerPage": 2000}

            response = requests.get(self.api_url, headers=self.headers, params=params)
            response.raise_for_status()
            process_batch(response.json())
            time.sleep(self.rate_limit_delay)

        # Fetch recently modified CVEs
        start_mod = latest_mod + timedelta(seconds=1)
        for cstart, cend in _chunk_ranges(start_mod, now):
            s = cstart.strftime("%Y-%m-%dT%H:%M:%S.000")
            e = cend.strftime("%Y-%m-%dT%H:%M:%S.000")
            params = {
                "lastModStartDate": s,
                "lastModEndDate": e,
                "resultsPerPage": 2000,
            }

            response = requests.get(self.api_url, headers=self.headers, params=params)
            response.raise_for_status()
            process_batch(response.json())
            time.sleep(self.rate_limit_delay)

        # Save updated dataset
        cleaned = list(existing_map.values())
        save_to_json(cleaned, self.history_path)
        self.logger.info(
            f"Incremental update complete: {new_count} new, {mod_count} modified CVEs"
        )
        return True

    def run(self) -> bool:
        """
        Execute appropriate fetch strategy.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.isfile(self.history_path):
                # No existing data - fetch everything
                return self.fetch_all_cves()
            else:
                # Existing data - perform incremental update
                return self.incremental_update()
        except Exception as e:
            self.logger.error(f"Run failed: {e}")
            return False
