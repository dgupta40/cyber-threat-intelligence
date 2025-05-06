#!/usr/bin/env python3
import os
import json
import logging
import argparse

from scraper.hackernews_scraper import HackerNewsScraper
from scraper.nvd_scraper       import NVDScraper
from scraper.krebsonsecurity_scraper import KrebsOnSecurityScraper

def scrape_hackernews(method: str):
    scraper = HackerNewsScraper(method=method)
    master = os.path.join(scraper.output_dir, 'hackernews.json')
    temp   = os.path.join(scraper.output_dir, 'hackernews_new.json')

    # load existing
    if os.path.exists(master):
        with open(master, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []

    seen_urls = {item['url'] for item in existing}

    # scrape into temp file
    all_items = scraper.scrape_new(temp)

    # filter out duplicates
    new_items = [i for i in all_items if i['url'] not in seen_urls]

    if new_items:
        combined = existing + new_items
        with open(master, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logging.info(f"📰 Added {len(new_items)} new HackerNews articles.")
    else:
        logging.info("📰 No new HackerNews articles found.")

    # cleanup
    try:
        os.remove(temp)
    except OSError:
        pass

def scrape_nvd(start_year: int, history_file: str):
    """
    NVDScraper.run() will:
      - on first run: pull everything since start_year → now and
        write a single history JSON at history_file
      - on subsequent runs: only pull new CVEs since last run
        and append them into that same history_file.
    """
    nvd = NVDScraper(start_year=start_year, history_file=history_file)
    if nvd.run():
        logging.info(f"🔍 NVD: up‑to‑date (history in {history_file})")
    else:
        logging.error("🔍 NVD: scrape failed")

# def merge_incremental(scraper, master_file, temp_file, key='url'):
#     if os.path.exists(master_file):
#         existing = json.load(open(master_file, 'r', encoding='utf-8'))
#     else:
#         existing = []
#     seen = {item[key] for item in existing}

#     all_items = scraper.scrape_new(temp_file)
#     new_items = [i for i in all_items if i[key] not in seen]

#     if new_items:
#         combined = existing + new_items
#         json.dump(combined, open(master_file, 'w', encoding='utf-8'),
#                   indent=2, ensure_ascii=False)
#         logging.info(f"Added {len(new_items)} new Krebs articles.")
#     else:
#         logging.info("No new Krebs articles found.")

#     try: os.remove(temp_file)
#     except: pass
def main():
    parser = argparse.ArgumentParser(description="Incremental HackerNews + NVD scraper")
    parser.add_argument('--method', choices=['scrapy', 'bs4'], default='scrapy',
                        help='Which method to use for HackerNews')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 1) HackerNews
    scrape_hackernews(args.method)

    # 2) NVD (back‑fill from 2019 on first run, then incremental)
    nvd = os.path.join('data','raw','nvd','nvd.json')
    os.makedirs(os.path.dirname(nvd), exist_ok=True)
    scrape_nvd(start_year=2019, history_file=nvd)

    # # 3) Kerbs on Security Blog
    # ks = KrebsOnSecurityScraper()
    # merge_incremental(
    #   scraper=ks,
    #   master_file=os.path.join(ks.output_dir, 'krebsonsecurity.json'),
    #   temp_file=os.path.join(ks.output_dir, 'krebsonsecurity_new.json'),
    #   key='url'
    # )

if __name__ == "__main__":
    main()
