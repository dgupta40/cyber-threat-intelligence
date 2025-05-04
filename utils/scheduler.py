from apscheduler.schedulers.background import BackgroundScheduler
import subprocess, sys, datetime, pathlib
from scraper import nvd_scraper

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY   = sys.executable

def _run_spider(spider_py):
    subprocess.run([PY, "-m", "scrapy", "runspider", str(spider_py)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def start():
    sched = BackgroundScheduler(timezone="UTC")

    # list of (id, callable) pairs ---------------------------
    jobs = {
        "hackernews":    lambda: _run_spider(ROOT/"scraper/hackernews_scraper.py"),
        "krebs":         lambda: _run_spider(ROOT/"scraper/krebsonsecurity_scraper.py"),
        "nvd":           lambda: nvd_scraper.pull_nvd(
                                 (datetime.datetime.utcnow()-datetime.timedelta(hours=6)).isoformat()+"Z",
                                 datetime.datetime.utcnow().isoformat()+"Z"),
    }

    now = datetime.datetime.utcnow()

    # immediate run *and* schedule every 6 h
    for jid, fn in jobs.items():
        fn()  # ---- run right away
        sched.add_job(fn, id=jid, trigger="interval", hours=6, next_run_time=now+datetime.timedelta(hours=6))

    sched.start()
