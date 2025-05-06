# # scraper/krebsonsecurity_scraper.py

# import os
# import json
# import logging
# from datetime import datetime
# import scrapy
# from scrapy.crawler import CrawlerProcess
# from utils.helpers import format_timestamp

# class KrebsOnSecuritySpider(scrapy.Spider):
#     name = "krebsonsecurity_spider"
#     start_urls = ['https://krebsonsecurity.com/']

#     def parse(self, response):
#         # updated selector for article links
#         links = response.css('article.post h2.entry-title a::attr(href)').getall()
#         for link in links:
#             yield response.follow(link, callback=self.parse_article)

#         # pagination
#         nxt = response.css('a.next.page-numbers::attr(href)').get()
#         if nxt:
#             yield response.follow(nxt, callback=self.parse)

#     def parse_article(self, response):
#         title = response.css('h1.entry-title::text').get()
#         # join all <p> under the entry-content container
#         content = " ".join(response.css('div.entry-content p::text').getall()).strip()

#         # date is in a <time> tag now
#         date_str = response.css('time.entry-date::text').get()
#         try:
#             dt = datetime.strptime(date_str.strip(), '%B %d, %Y')
#             date = dt.strftime('%Y-%m-%d')
#         except:
#             date = format_timestamp()

#         # categories live under .cat-links
#         categories = response.css('span.cat-links a::text').getall()

#         yield {
#             'source':     'krebsonsecurity',
#             'title':      title,
#             'content':    content,
#             'date':       date,
#             'categories': categories,
#             'url':        response.url
#         }


# class KrebsOnSecurityScraper:
#     def __init__(self):
#         self.output_dir = 'data/raw/krebsonsecurity'
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.logger = logging.getLogger(__name__)

#     def _run_spider(self, output_path):
#         """Run the Scrapy spider and write to the given output_path, then return loaded items."""
#         self.logger.info(f"Running KrebsOnSecuritySpider → {output_path}")
#         process = CrawlerProcess(settings={
#             'FEEDS': {
#                 output_path: {'format': 'json', 'overwrite': True}
#             },
#             'LOG_LEVEL': 'ERROR',
#             'USER_AGENT': 'Mozilla/5.0 (compatible; Bot/1.0)'
#         })
#         process.crawl(KrebsOnSecuritySpider)
#         process.start()  # blocks until finished

#         # read back in
#         with open(output_path, 'r', encoding='utf-8') as f:
#             return json.load(f)

#     def scrape_new(self, temp_path):
#         """
#         Incremental interface: scrape into temp_path and return the list of items.
#         """
#         return self._run_spider(temp_path)

#     def scrape_all(self):
#         """
#         One‑off full dump (timestamped file).
#         """
#         ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#         out = os.path.join(self.output_dir, f'krebsonsecurity_{ts}.json')
#         self._run_spider(out)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s %(levelname)s %(message)s')
#     KrebsOnSecurityScraper().scrape_all()
