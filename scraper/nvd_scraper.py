import os
import logging
import json
import gzip
import time
from datetime import datetime, timedelta
import requests

from utils.helpers import save_to_json, safe_request

class NVDScraper:
    """Fetch and maintain a master CVE history from NVD."""
    
    def __init__(self,
                 start_year: int = 2019,
                 history_file: str = 'data/raw/nvd/nvd.json'):
        self.output_dir    = 'data/raw/nvd'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger        = logging.getLogger(__name__)
        
        # NVD endpoints
        self.json_feed_template = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{}.json.gz"
        self.api_url             = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        
        # where we keep our ever-growing history
        self.history_path = history_file
        self.start_year   = start_year
        
        # optional API key (for higher rate‐limits)
        self.api_key = os.getenv('NVD_API_KEY', None)
    
    def backfill_history(self):
        """
        ONE‐TIME: Download and combine year‐by‐year JSON feeds
        from start_year through current year.
        """
        self.logger.info(f"Backfilling CVEs from {self.start_year} to today")
        cve_dict = {}
        
        for year in range(self.start_year, datetime.utcnow().year + 1):
            url  = self.json_feed_template.format(year)
            gzfp = os.path.join(self.output_dir, f'nvdcve-{year}.json.gz')
            
            # download
            r = safe_request(url, stream=True)
            if not r:
                self.logger.error(f"Failed to download {url}")
                continue
            with open(gzfp, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            
            # extract & load
            with gzip.open(gzfp, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get('CVE_Items', [])
            self.logger.info(f"  year={year} → {len(items)} items")
            
            # dedupe into dict by CVE ID
            for item in items:
                cve_id = (item.get('cve', {}) \
                               .get('CVE_data_meta', {}) \
                               .get('ID'))
                if cve_id:
                    cve_dict[cve_id] = item
        
        # write out
        history = {
            'generated': datetime.utcnow().isoformat(),
            'total_cves': len(cve_dict),
            'CVE_Items': list(cve_dict.values())
        }
        save_to_json(history, self.history_path)
        self.logger.info(f"Backfill complete: {len(cve_dict)} unique CVEs → {self.history_path}")
    
    def incremental_update(self):
        """
        Fetch only CVEs published since the newest one in history,
        append and rewrite the history file.
        """
        # load existing
        with open(self.history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        existing = { item['cve']['CVE_data_meta']['ID']: item
                     for item in history.get('CVE_Items', []) }
        self.logger.info(f"History contains {len(existing)} CVEs")
        
        # find the latest published date in history
        latest_dt = max(
            datetime.fromisoformat(
                item['publishedDate'].rstrip('Z')
            )
            for item in existing.values()
            if 'publishedDate' in item
        )
        pub_start = (latest_dt + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000")
        pub_end   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000")
        
        self.logger.info(f"Fetching incremental CVEs: {pub_start} → {pub_end}")
        
        # call the API
        params = {
            'pubStartDate': pub_start,
            'pubEndDate':   pub_end,
            'resultsPerPage': 2000
        }
        headers = {}
        if self.api_key:
            headers['apiKey'] = self.api_key
        
        r = requests.get(self.api_url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        total = data.get('totalResults', 0)
        self.logger.info(f"API reports {total} new/updated CVEs")
        
        # fetch in pages
        new_count = 0
        idx = 0
        while idx < total:
            params['startIndex'] = idx
            batch = requests.get(self.api_url, params=params, headers=headers).json()
            vulns = batch.get('vulnerabilities', [])
            for v in vulns:
                cve = v.get('cve')
                if not cve:
                    continue
                cve_id = cve.get('CVE_data_meta', {}).get('ID')
                if cve_id and cve_id not in existing:
                    existing[cve_id] = v
                    new_count += 1
            idx += len(vulns)
            time.sleep(0.6)
        
        self.logger.info(f"Appending {new_count} new CVEs to history")
        
        # rewrite full history
        out = {
            'generated': datetime.utcnow().isoformat(),
            'total_cves': len(existing),
            'CVE_Items': list(existing.values())
        }
        save_to_json(out, self.history_path)
        self.logger.info(f"History updated: now {len(existing)} CVEs in {self.history_path}")
    
    def run(self):
        """
        On first run, do a backfill; otherwise do an incremental update.
        """
        try:
            if not os.path.exists(self.history_path):
                self.backfill_history()
            else:
                self.incremental_update()
            return True
        except Exception as e:
            self.logger.error(f"Run failed: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    scraper = NVDScraper()
    scraper.run()
