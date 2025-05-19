#!/usr/bin/env python3
"""
scheduler.py: Automate periodic scraping of NVD and The Hacker News.
Run this script to schedule scrapes every 6 hours.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from scraper.hackernews_scraper import HackerNewsScraper
from scraper.nvd_scraper import NVDScraper

# Paths to history files (ensure these match your project structure)
HN_HISTORY = 'data/raw/hackernews/hackernews.json'
NVD_HISTORY = 'data/raw/nvd/nvd_cleaned.json'


def scrape_hackernews():
    """Run incremental scrape for The Hacker News."""
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] Starting HackerNews scrape...")
    scraper = HackerNewsScraper(history_file=HN_HISTORY)
    success = scraper.run()
    status = "succeeded" if success else "failed"
    print(f"[{datetime.utcnow().isoformat()}] HackerNews scrape {status}.")


def scrape_nvd():
    """Run incremental update for NVD CVE data."""
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] Starting NVD incremental update...")
    scraper = NVDScraper(history_file=NVD_HISTORY)
    success = scraper.incremental_update()
    status = "succeeded" if success else "failed"
    print(f"[{datetime.utcnow().isoformat()}] NVD update {status}.")


if __name__ == '__main__':
    scheduler = BlockingScheduler(timezone='UTC')
    now = datetime.utcnow()

    # Schedule both scrapers: run at start, then every 6 hours
    scheduler.add_job(
        scrape_hackernews,
        trigger='interval',
        hours=6,
        next_run_time=now
    )
    scheduler.add_job(
        scrape_nvd,
        trigger='interval',
        hours=6,
        next_run_time=now
    )

    print(f"[{datetime.utcnow().isoformat()}] Scheduler started: scraping every 6 hours.")
    scheduler.start()
