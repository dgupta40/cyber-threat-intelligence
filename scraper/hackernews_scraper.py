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
    start_urls = ["https://thehackernews.com/search/label/Vulnerability"]
    custom_settings = {
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 15,
        "LOG_LEVEL": "ERROR",
        "USER_AGENT": "Mozilla/5.0 (compatible; HackerNewsBot/1.0)",
    }

    def parse(self, response):
        # Follow each article link
        for link in response.css("a.story-link::attr(href)").getall():
            yield response.follow(link, callback=self.parse_article)
        # Follow pagination until no next page
        next_page = response.css("a.blog-pager-older-link-mobile::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        # Archive raw HTML for future re-parsing
        raw_dir = "data/raw/hackernews/html"
        os.makedirs(raw_dir, exist_ok=True)
        url_hash = hashlib.sha256(response.url.encode()).hexdigest()
        with open(
            os.path.join(raw_dir, f"{url_hash}.html"), "w", encoding="utf-8"
        ) as f:
            f.write(response.text)

        # Timestamp when scraped
        ingest_ts_thn = datetime.utcnow().isoformat()

        # Parse publication date with various formats
        date = None
        date_text = (
            response.xpath("//meta[@property='article:published_time']/@content").get()
            or ""
        )
        for pat in ("%Y-%m-%dT%H:%M:%SZ", "%B %d, %Y", "%b %d, %Y"):
            try:
                dt = datetime.strptime(date_text.strip(), pat)
                date = dt.strftime("%Y-%m-%d")
                break
            except Exception:
                continue

        # Fallback regex extraction for non-meta dates
        if not date:
            match = re.search(r"([A-Za-z]{3,9} \d{1,2},\s*\d{4})", response.text)
            if match:
                dt_str = match.group(1).strip()
                for pat in ("%B %d, %Y", "%b %d, %Y"):
                    try:
                        dt = datetime.strptime(dt_str, pat)
                        date = dt.strftime("%Y-%m-%d")
                        break
                    except Exception:
                        continue

        if not date:
            self.logger.warning(
                f"Unparseable date on {response.url}, defaulting to current date"
            )
            date = datetime.utcnow().strftime("%Y-%m-%d")

        # Title
        full_title = response.xpath("//title/text()").get(default="").strip()
        title = full_title.split("–")[0].strip()

        # Content paragraphs
        paragraphs = response.css("div.articlebody p::text").getall()
        body = " ".join(p.strip() for p in paragraphs if p.strip())

        # Full text of the page for more comprehensive extraction
        full_text = response.xpath("//text()").getall()
        full_text = " ".join(full_text)

        # Prepend title to content for richer text field
        text = f"{title}. {body}"

        # Extract CVE IDs, normalized
        raw_cves = re.findall(r"(CVE-\d{4}-\d{4,6})", full_text, flags=re.IGNORECASE)
        cves = [cve.upper() for cve in raw_cves]

        # Extract affected products
        products = []
        # Common product patterns
        product_patterns = [
            r"affects?\s+([A-Za-z0-9\s\-\.]+)\s+version\s*([0-9\.\-\s]+)",
            r"([A-Za-z0-9\s\-\.]+)\s+version\s*([0-9\.\-\s]+)\s+(?:is\s+)?(?:vulnerable|affected)",
            r"vulnerable\s+versions?\s+of\s+([A-Za-z0-9\s\-\.]+)",
            r"([A-Za-z0-9\s\-\.]+)\s+(?:versions?\s*)?(?:before|prior\s+to|up\s+to)\s+([0-9\.\-\s]+)",
        ]

        for pattern in product_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                product_name = match.group(1).strip()
                if len(match.groups()) > 1:
                    version = match.group(2).strip()
                    products.append(f"{product_name} {version}")
                else:
                    products.append(product_name)

        # Also look for standalone product mentions near vulnerability keywords
        if not products:
            # Common software names that might be mentioned
            common_products = [
                "Windows",
                "Linux",
                "Android",
                "iOS",
                "Chrome",
                "Firefox",
                "Safari",
                "Edge",
                "Apache",
                "Nginx",
                "WordPress",
                "Drupal",
                "Jenkins",
                "Docker",
                "Kubernetes",
                "OpenSSL",
                "Java",
                "Python",
                "PHP",
                "MySQL",
                "PostgreSQL",
                "MongoDB",
                "Redis",
                "Elasticsearch",
            ]

            for product in common_products:
                if re.search(rf"\b{product}\b", full_text, re.IGNORECASE):
                    # Check if it's in context of vulnerability
                    context_pattern = rf"(?:vulnerability|flaw|bug|exploit|affected|vulnerable)[\s\S]{{0,50}}\b{product}\b"
                    if re.search(context_pattern, full_text, re.IGNORECASE):
                        products.append(product)

        # Remove duplicates and clean up
        products = list(set(p.strip() for p in products if p.strip()))

        # Extract tags
        tags = response.css("span.categ a span::text").getall()

        yield {
            "source": "The Hacker News",
            "text": text,
            "published_date": date,
            "products": products,
            "tags": tags,
            "url": response.url,
            "cves": cves,
            "ingest_ts_thn": ingest_ts_thn,
        }


class HackerNewsScraper:
    """Scraper with incremental update for The Hacker News vulnerability label using Scrapy."""

    # Updated keys without cvss_score, severity, and cwe
    REQUIRED_KEYS = {
        "source",
        "text",
        "published_date",
        "products",
        "tags",
        "url",
        "cves",
        "ingest_ts_thn",
    }

    def __init__(self, history_file: str = "data/raw/hackernews/hackernews.json"):
        self.output_dir = os.path.dirname(history_file)
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_file = history_file
        self.logger = logging.getLogger(__name__)

    def scrape_new(self, temp_file: str) -> list[dict]:
        # Remove leftover temp file to ensure clean overwrite
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e:
                self.logger.warning(
                    f"Could not remove existing temp file {temp_file}: {e}"
                )

        # Configure feed export
        feed_opts = {"format": "json", "overwrite": True}
        settings = {"FEEDS": {temp_file: feed_opts}, **HackerNewsSpider.custom_settings}

        process = CrawlerProcess(settings=settings)
        process.crawl(HackerNewsSpider)
        process.start()

        # Load the freshly written JSON array
        with open(temp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def run(self) -> bool:
        temp_file = os.path.join(self.output_dir, "_hn_temp.json")
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
        existing_urls = {a["url"] for a in existing}
        new_articles = [a for a in valid if a["url"] not in existing_urls]

        if not new_articles:
            self.logger.info("No new articles to add.")
            return True

        # Merge and sort by published_date (newest first)
        combined = new_articles + existing
        combined.sort(key=lambda x: x.get("published_date", ""), reverse=True)

        save_to_json(combined, self.history_file)
        self.logger.info(
            f"Added {len(new_articles)} new articles; total now {len(combined)}."
        )
        return True
