import os
import logging
import json
import time
from datetime import datetime, timedelta
import requests

from utils.helpers import save_to_json

# NVD 2.0 API restricts date ranges to a maximum of 120 days per request
def _chunk_ranges(start: datetime, end: datetime, max_days: int = 120):
    """
    Yield (chunk_start, chunk_end) pairs covering [start..end]
    without exceeding max_days in any interval.
    """
    delta = timedelta(days=max_days)
    chunk_start = start
    while chunk_start < end:
        # subtract 1 millisecond so intervals butt up against each other
        chunk_end = min(end, chunk_start + delta - timedelta(milliseconds=1))
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(milliseconds=1)

class NVDScraper:
    """Fetch and maintain a master CVE history from NVD using API 2.0 with 120-day chunking."""
    def __init__(self,
                 start_year: int = 2019,
                 history_file: str = 'data/raw/nvd/nvd.json'):
        self.output_dir = os.path.dirname(history_file)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.api_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.history_path = history_file
        self.start_year = start_year

        self.api_key = os.getenv('NVD_API_KEY')
        self.headers = {}
        if self.api_key:
            self.headers['apiKey'] = self.api_key
            self.rate_limit_delay = 0.6  # 50 requests per 30 seconds
        else:
            self.rate_limit_delay = 6.0  # 5 requests per 30 seconds
            self.logger.warning("No NVD_API_KEY found. Rate limits will be lower.")

    def fetch_all_cves(self) -> bool:
        """
        Full fetch: iterate each year from start_year to today,
        split into <=120-day chunks, and aggregate all CVEs.
        """
        self.logger.info(f"Fetching all CVEs from {self.start_year} to today via API 2.0")
        cve_dict: dict[str, dict] = {}
        current_year = datetime.utcnow().year

        for year in range(self.start_year, current_year + 1):
            year_start = datetime(year, 1, 1)
            year_end = datetime.utcnow() if year == current_year else datetime(year, 12, 31, 23, 59, 59, 999000)

            for chunk_start, chunk_end in _chunk_ranges(year_start, year_end):
                s = chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000")
                e = chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000")
                params = {
                    'pubStartDate': s,
                    'pubEndDate':   e,
                    'resultsPerPage': 2000
                }
                self.logger.info(f"Fetching CVEs for {s} -> {e}")

                response = requests.get(self.api_url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                total = data.get('totalResults', 0)
                if total == 0:
                    continue

                self._process_batch(data, cve_dict)
                idx = len(data.get('vulnerabilities', []))
                while idx < total:
                    params['startIndex'] = idx
                    batch_resp = requests.get(self.api_url, headers=self.headers, params=params)
                    batch_resp.raise_for_status()
                    batch = batch_resp.json()
                    self._process_batch(batch, cve_dict)
                    idx += len(batch.get('vulnerabilities', []))
                    time.sleep(self.rate_limit_delay)

            self.logger.info(f"Year {year} complete: Total CVEs so far: {len(cve_dict)}")

        if not cve_dict:
            self.logger.error("No CVEs were found or processed!")
            return False

        history = {
            'generated': datetime.utcnow().isoformat(),
            'total_cves': len(cve_dict),
            'format': 'NVD_CVE',
            'version': '2.0',
            'vulnerabilities': list(cve_dict.values())
        }
        save_to_json(history, self.history_path)
        self.logger.info(f"Fetch complete: {len(cve_dict)} unique CVEs -> {self.history_path}")
        return True

    def incremental_update(self) -> bool:
        """
        Incremental: load existing history, then fetch only new or modified CVEs
        since the latest timestamps, using <=120-day chunking if needed.
        """
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading history file: {e}")
            self.logger.info("Starting fresh fetch")
            return self.fetch_all_cves()

        existing_cves = {
            v['cve']['id']: v
            for v in history.get('vulnerabilities', [])
            if v.get('cve', {}).get('id')
        }
        if not existing_cves:
            self.logger.error("No valid CVEs found in history file")
            return self.fetch_all_cves()

        # Determine latest published/modified datetimes
        latest_pub = datetime(2000,1,1)
        latest_mod = datetime(2000,1,1)
        for v in existing_cves.values():
            c = v['cve']
            try:
                p = datetime.fromisoformat(c.get('published','').rstrip('Z'))
                latest_pub = max(latest_pub, p)
            except: pass
            try:
                m = datetime.fromisoformat(c.get('lastModified','').rstrip('Z'))
                latest_mod = max(latest_mod, m)
            except: pass

        new_count = 0
        mod_count = 0
        now = datetime.utcnow()

        # Fetch new publications
        start_pub = latest_pub + timedelta(seconds=1)
        for cstart, cend in _chunk_ranges(start_pub, now):
            self.logger.info(f"Checking for new CVEs published {cstart} -> {cend}")
            params = {
                'pubStartDate': cstart.strftime("%Y-%m-%dT%H:%M:%S.000"),
                'pubEndDate':   cend.strftime("%Y-%m-%dT%H:%M:%S.000"),
                'resultsPerPage': 2000
            }
            new_count += self._fetch_and_update(params, existing_cves)
            time.sleep(self.rate_limit_delay)

        # Fetch recent modifications
        start_mod = latest_mod + timedelta(seconds=1)
        for cstart, cend in _chunk_ranges(start_mod, now):
            self.logger.info(f"Checking for CVEs modified {cstart} -> {cend}")
            params = {
                'lastModStartDate': cstart.strftime("%Y-%m-%dT%H:%M:%S.000"),
                'lastModEndDate':   cend.strftime("%Y-%m-%dT%H:%M:%S.000"),
                'resultsPerPage': 2000
            }
            mod_count += self._fetch_and_update(params, existing_cves, update_existing=True)
            time.sleep(self.rate_limit_delay)

        history = {
            'generated': datetime.utcnow().isoformat(),
            'total_cves': len(existing_cves),
            'format': 'NVD_CVE',
            'version': '2.0',
            'vulnerabilities': list(existing_cves.values())
        }
        save_to_json(history, self.history_path)
        self.logger.info(f"Incremental update complete: {new_count} new, {mod_count} modified CVEs")
        return True

    def _process_batch(self, data: dict, cve_dict: dict):
        """Process a single page of API response, adding new CVEs to cve_dict."""
        for v in data.get('vulnerabilities', []):
            cid = v.get('cve', {}).get('id')
            if cid:
                cve_dict[cid] = v

    def _fetch_and_update(self, params: dict, existing: dict, update_existing: bool=False) -> int:
        """
        Fetch one paginated date-chunk and update existing dict.
        """
        count = 0
        try:
            resp = requests.get(self.api_url, headers=self.headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            total = data.get('totalResults', 0)
            if not total:
                return 0
            count += self._update_cves(data, existing, update_existing)
            idx = len(data.get('vulnerabilities', []))
            while idx < total:
                params['startIndex'] = idx
                batch_resp = requests.get(self.api_url, headers=self.headers, params=params)
                batch_resp.raise_for_status()
                batch = batch_resp.json()
                processed = self._update_cves(batch, existing, update_existing)
                count += processed
                idx += len(batch.get('vulnerabilities', []))
                self.logger.info(f"Progress: {idx}/{total}, +{processed}")
                time.sleep(self.rate_limit_delay)
        except Exception as e:
            self.logger.error(f"Error fetching/updating CVEs chunk: {e}")
        return count

    def _update_cves(self, data: dict, existing: dict, update_existing: bool) -> int:
        """Add new or update existing CVEs from one page of results."""
        count = 0
        for v in data.get('vulnerabilities', []):
            cid = v.get('cve', {}).get('id')
            if not cid:
                continue
            if cid not in existing:
                existing[cid] = v
                count += 1
            elif update_existing:
                old = existing[cid]['cve'].get('lastModified', '')
                new = v['cve'].get('lastModified', '')
                if new > old:
                    existing[cid] = v
                    count += 1
        return count

    def run(self) -> bool:
        """Entry point—full fetch if no history, else incremental update."""
        try:
            if not os.path.isfile(self.history_path):
                return self.fetch_all_cves()
            return self.incremental_update()
        except Exception as e:
            self.logger.error(f"Run failed: {e}")
            return False
