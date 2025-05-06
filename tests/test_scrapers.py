from pathlib import Path
from scraper import hackernews_scraper, krebsonsecurity_scraper

def test_hn_scraper(tmp_path, monkeypatch):
    # run with pages=1 & write to tmp dir
    monkeypatch.setattr("utils.helpers.RAW_DIR", tmp_path)
    hackernews_scraper.scrape(pages=1)
    assert any(tmp_path.glob("hackernews_*.json.gz"))

def test_krebs_scraper(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.helpers.RAW_DIR", tmp_path)
    krebsonsecurity_scraper.scrape(pages=1)
    assert any(tmp_path.glob("krebsonsecurity_*.json.gz"))
