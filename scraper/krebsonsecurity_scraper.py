"""
Scrape Krebs on Security front page + older-posts pagination.
• Uses the h2.entry-title link list you verified in Colab
• Each article URL pattern:  /YYYY/MM/DD/…
"""

import datetime, time, re
from bs4 import BeautifulSoup
from utils.helpers import save_raw, safe_request

SITE = "https://krebsonsecurity.com"
ARTICLE_RE = re.compile(r"https://krebsonsecurity\.com/\d{4}/\d{2}/\d{2}/")

def _scrape_page(url):
    soup = BeautifulSoup(safe_request(url).text, "html.parser")

    for h2 in soup.select("h2.entry-title"):
        a = h2.find("a", href=True)
        if not a:
            continue
        link = a["href"]
        if not ARTICLE_RE.match(link):
            continue

        art = BeautifulSoup(safe_request(link).text, "html.parser")
        title_tag = art.select_one("h1.entry-title")
        body_div  = art.select_one("div.entry-content")
        if not (title_tag and body_div):
            continue

        item = {
            "source": "krebsonsecurity",
            "url": link,
            "title": title_tag.get_text(strip=True),
            "published": art.select_one("time.entry-date")["datetime"],
            "body": "\n".join(p.get_text(" ", strip=True) for p in body_div.select("p")),
            "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        save_raw(item, "krebsonsecurity")
        print("✅ saved:", link)

    older = soup.select_one("a.older-posts")     # pagination link
    return older["href"] if older else None

def scrape(pages: int = 2):
    url = SITE
    for _ in range(pages):
        url = _scrape_page(url)
        if not url:
            break
        time.sleep(1)

if __name__ == "__main__":
    scrape()
