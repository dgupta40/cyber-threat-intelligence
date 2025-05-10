import os
import re
import json
import logging
import hashlib
from datetime import datetime

import scrapy
from scrapy.crawler import CrawlerProcess

from utils.helpers import save_to_json, load_from_json


class HackerNewsSpider(scrapy.Spider):
    name = "hackernews_spider"
    start_urls = ['https://thehackernews.com/search/label/Vulnerability']
    custom_settings = {
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 3,
        'DOWNLOAD_TIMEOUT': 15,
        'LOG_LEVEL': 'ERROR',
        'USER_AGENT': 'Mozilla/5.0 (compatible; HackerNewsBot/1.0)',
    }

    def parse(self, response):
        # Follow each article link
        for link in response.css('a.story-link::attr(href)').getall():
            yield response.follow(link, callback=self.parse_article)
        # Follow pagination until no next page
        next_page = response.css('a.blog-pager-older-link-mobile::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        # Archive raw HTML for future re-parsing
        raw_dir = 'data/raw/hackernews/html'
        os.makedirs(raw_dir, exist_ok=True)
        url_hash = hashlib.sha256(response.url.encode()).hexdigest()
        with open(os.path.join(raw_dir, f'{url_hash}.html'), 'w', encoding='utf-8') as f:
            f.write(response.text)

        # Timestamp when scraped
        ingest_ts_thn = datetime.utcnow().isoformat()

        # Parse publication date
        date = None
        date_text = response.xpath("//meta[@property='article:published_time']/@content").get() or ''
        for pat in ('%Y-%m-%dT%H:%M:%SZ', '%B %d, %Y'):
            try:
                dt = datetime.strptime(date_text.strip(), pat)
                date = dt.strftime('%Y-%m-%d')
                break
            except Exception:
                continue
        if not date:
            m = re.search(r'([A-Za-z]+ \d{1,2},\s*\d{4})', response.text)
            if m:
                dt = datetime.strptime(m.group(1).strip(), '%B %d, %Y')
                date = dt.strftime('%Y-%m-%d')
        if not date:
            self.logger.warning(f"Unparseable date on {response.url}")
            date = datetime.utcnow().strftime('%Y-%m-%d')

        # Title
        full_title = response.xpath('//title/text()').get(default='').strip()
        title = full_title.split('–')[0].strip()

        # Content paragraphs
        paragraphs = response.css('div.articlebody p::text').getall()
        body = " ".join(p.strip() for p in paragraphs if p.strip())

        # Prepend title to content for richer text field
        text = f"{title}. {body}"

        # Extract CVE IDs, normalized
        raw_cves = re.findall(r'(CVE-\d{4}-\d{4,6})', body, flags=re.IGNORECASE)
        cves = [cve.upper() for cve in raw_cves]

        # Extract tags
        tags = response.css('span.categ a span::text').getall()

        yield {
            'source': 'The Hacker News',
            'text': text,
            'published_date': date,
            'tags': tags,
            'url': response.url,
            'cves': cves,
            'ingest_ts_thn': ingest_ts_thn
        }


class HackerNewsScraper:
    """Scraper with incremental update for TheHackerNews vulnerability label using Scrapy."""

    REQUIRED_KEYS = {
        'source', 'title', 'content',
        'date', 'tags', 'url',
        'cves', 'ingest_ts_thn'
    }

    def __init__(self, history_file: str = 'data/raw/hackernews/hackernews.json'):
        self.output_dir = os.path.dirname(history_file)
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_file = history_file
        self.logger = logging.getLogger(__name__)

    def scrape_new(self, temp_file: str) -> list[dict]:
        """
        Run Scrapy spider and dump JSON feed to temp_file (fresh each time).
        """
        # Remove leftover temp file to ensure clean overwrite
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e:
                self.logger.warning(f"Could not remove existing temp file {temp_file}: {e}")

        # Tell Scrapy to overwrite the feed file
        feed_opts = {'format': 'json', 'overwrite': True}
        settings = {'FEEDS': {temp_file: feed_opts}, **HackerNewsSpider.custom_settings}

        process = CrawlerProcess(settings=settings)
        process.crawl(HackerNewsSpider)
        process.start()

        # Load the freshly written JSON array
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def run(self) -> bool:
        """
        Perform incremental scraping and merge into history_file.
        """
        temp_file = os.path.join(self.output_dir, '_hn_temp.json')
        scraped = self.scrape_new(temp_file)

        try:
            os.remove(temp_file)
        except OSError:
            pass

        # Validate scraped items
        valid = []
        for art in scraped:
            missing = self.REQUIRED_KEYS - set(art.keys())
            if missing:
                self.logger.error(f"Article missing fields {missing}: {art.get('url')}")
            else:
                valid.append(art)

        # Load existing history
        try:
            existing = load_from_json(self.history_file)
            if not isinstance(existing, list):
                self.logger.warning("Existing history is not a list; resetting.")
                existing = []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load history ({e}); starting fresh.")
            existing = []

        # Deduplicate by URL
        existing_urls = {a['url'] for a in existing}
        new_articles = [a for a in valid if a['url'] not in existing_urls]

        if not new_articles:
            self.logger.info("No new articles to add.")
            return True

        # Merge and sort by date (newest first)
        combined = new_articles + existing
        combined.sort(key=lambda x: x.get('date', ''), reverse=True)

        save_to_json(combined, self.history_file)
        self.logger.info(f"Added {len(new_articles)} new articles; total now {len(combined)}.")
        return True
