"""
Scrape the front page of The Hacker News and older pages if desired.
• Primary selector:  div.body-post  (as in your Colab)
• Fallback: scan all links for /YYYY/MM/ pattern when few articles are found
• De-duplicates via utils.save_raw()
"""

import datetime, time, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.helpers import save_raw, safe_request

BASE = "https://thehackernews.com/search/label/Vulnerability"
ARTICLE_RE = re.compile(r"https://thehackernews\.com/\d{4}/\d{2}/")

def _extract_article(el):
    """Pull title, link, date, description from a div.body-post block."""
    story_link = el.select_one("a.story-link")
    if not story_link:
        return None

    link = story_link["href"]
    if not ARTICLE_RE.match(link):
        return None                      # skip promo links

    # title can be in img alt or other tags
    img = el.select_one("img")
    title = (img.get("alt") or "").strip() if img else ""
    if not title:
        cand = el.select_one(".home-title, .story-title, h1, h2, h3")
        title = cand.get_text(strip=True) if cand else None
    if not title:
        return None

    date_el = el.select_one(".item-label, .story-time, time")
    date = date_el.get_text(strip=True) if date_el else None

    desc_el = el.select_one(".home-desc, .story-excerpt, p")
    description = desc_el.get_text(strip=True) if desc_el else None

    return {
        "source": "hackernews",
        "url": link,
        "title": title,
        "published": date,
        "body": description,
        "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

def _scrape_frontpage():
    soup = BeautifulSoup(
        safe_request(BASE).text,
        "html.parser",
    )
    articles = []
    for post in soup.select("div.body-post"):
        art = _extract_article(post)
        if art:
            save_raw(art, "hackernews")
            articles.append(art["url"])
    return articles, soup

def _fallback_scan(soup, already_saved):
    """Scan all <a> tags for missed article links (rare)."""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href in already_saved or not ARTICLE_RE.match(href):
            continue
        title = (a.text or a.get("title") or "").strip()
        if len(title) < 10:
            continue
        art = {
            "source": "hackernews",
            "url": href,
            "title": title,
            "published": None,
            "body": None,
            "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        save_raw(art, "hackernews")

def scrape(pages: int = 2):        # <— crawl first 2 pages by default
    print(f"🔎  Hacker News – Vulnerability (pages={pages})")
    url = BASE
    already_saved = set()
    for p in range(pages):
        saved, soup = _scrape_frontpage() if p == 0 else _scrape_page(url)
        already_saved.update(saved)

        next_link = soup.select_one("a.blog-pager-older-link")
        if not next_link:
            break
        url = next_link["href"]
        time.sleep(1)              # polite delay

def _scrape_page(url):
    soup = BeautifulSoup(safe_request(url).text, "html.parser")
    articles = []
    for post in soup.select("div.body-post"):
        art = _extract_article(post)
        if art:
            save_raw(art, "hackernews")
            articles.append(art["url"])
    return articles, soup


if __name__ == "__main__":
    scrape()
