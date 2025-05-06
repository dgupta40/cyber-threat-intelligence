import os
import re
import json
import logging
from datetime import datetime

import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup

from utils.helpers import format_timestamp, safe_request

class HackerNewsSpider(scrapy.Spider):
    name = "hackernews_spider"
    start_urls = ['https://thehackernews.com/search/label/Vulnerability']

    def parse(self, response):
        article_links = response.css('a.story-link::attr(href)').getall()
        self.logger.info(f"Found {len(article_links)} article links on list page")
        for link in article_links:
            yield response.follow(link, callback=self.parse_article)

        next_page = response.css('a.blog-pager-older-link-mobile::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        # 1. Title
        full_title = response.xpath('//title/text()').get(default='').strip()
        title = full_title.split('–')[0].strip()

        # 2. Date
        html = response.text
        m = re.search(r'([A-Za-z]+ \d{1,2}, \s*\d{4})', html)
        if m:
            try:
                dt = datetime.strptime(m.group(1).strip(), '%B %d, %Y')
                date = dt.strftime('%Y-%m-%d')
            except Exception:
                date = format_timestamp()
        else:
            date = format_timestamp()

        # 3. Content
        paragraphs = response.css('div.articlebody p::text').getall()
        content = " ".join(p.strip() for p in paragraphs if p.strip())

        # 4. CVEs
        cves = re.findall(r'CVE-\d{4}-\d{4,6}', content)

        # 5. Patched versions
        patched_versions = []
        for ul in response.css('div.articlebody ul'):
            txt = " ".join(ul.css('li::text').getall())
            if any(os in txt for os in ['iOS', 'macOS', 'tvOS', 'visionOS']):
                patched_versions = [li.strip() for li in ul.css('li::text').getall()]

        # 6. YouTube embeds
        videos = response.css('div.articlebody iframe::attr(src)').re(r'.*youtube\.com.*')

        # 7. Tags
        tags = response.css('span.categ a span::text').getall()

        yield {
            'source': 'thehackernews',
            'title': title,
            'content': content,
            'date': date,
            'tags': tags,
            'url': response.url,
            'cves': cves,
            'patched_versions': patched_versions,
            'videos': videos
        }


class HackerNewsScraper:
    def __init__(self, method="scrapy"):
        self.output_dir = 'data/raw/hackernews'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.method = method

    def scrape_new(self, output_file):
        """
        Scrape into `output_file`, return list of all scraped items.
        """
        if self.method == "bs4":
            articles = self._scrape_with_requests()
            # write them out so main.py can load the same file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            return articles
        else:
            # Scrapy path
            self._scrape_with_scrapy(output_file)
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _scrape_with_scrapy(self, output_file):
        """
        Run the Scrapy spider and dump a JSON feed to `output_file`.
        """
        process = CrawlerProcess(settings={
            'FEEDS': {
                output_file: {'format': 'json'},
            },
            'LOG_LEVEL': 'ERROR',
            'USER_AGENT': 'Mozilla/5.0 (compatible; HackerNewsBot/1.0)'
        })
        process.crawl(HackerNewsSpider)
        process.start()

    def _scrape_with_requests(self):
        """
        Pull pages via requests/BS4, return list of article dicts.
        """
        from utils.helpers import format_timestamp  # in case it got overwritten
        base_url = 'https://thehackernews.com/search/label/Vulnerability'
        articles = []
        page_url = base_url

        for _ in range(5):
            resp = safe_request(page_url)
            if not resp:
                break
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = [a['href'] for a in soup.select('a.story-link')]
            for link in links:
                art = self._parse_article_with_bs4(link)
                if art:
                    articles.append(art)
            np = soup.select_one('a.blog-pager-older-link-mobile')
            if np:
                page_url = np['href']
            else:
                break

        return articles

    def _parse_article_with_bs4(self, url):
        resp = safe_request(url)
        if not resp:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Title
        title_tag = soup.find('title')
        title = title_tag.text.split('–')[0].strip() if title_tag else ''

        # Date
        m = re.search(r'([A-Za-z]+ \d{1,2}, \s*\d{4})', resp.text)
        if m:
            try:
                dt = datetime.strptime(m.group(1).strip(), '%B %d, %Y')
                date = dt.strftime('%Y-%m-%d')
            except:
                date = format_timestamp()
        else:
            date = format_timestamp()

        # Content
        paras = soup.select('div.articlebody p')
        content = " ".join(p.get_text(strip=True) for p in paras)

        # CVEs
        cves = re.findall(r'CVE-\d{4}-\d{4,6}', content)

        # Patched versions
        patched = []
        for ul in soup.select('div.articlebody ul'):
            txt = ul.get_text()
            if any(os in txt for os in ['iOS', 'macOS', 'tvOS', 'visionOS']):
                patched = [li.get_text(strip=True) for li in ul.select('li')]

        # Videos
        videos = [ifr['src'] for ifr in soup.select('div.articlebody iframe[src*="youtube.com"]')]

        # Tags
        tags = [t.get_text(strip=True) for t in soup.select('span.categ a span')]

        return {
            'source': 'thehackernews',
            'title': title,
            'content': content,
            'date': date,
            'tags': tags,
            'url': url,
            'cves': cves,
            'patched_versions': patched,
            'videos': videos
        }
