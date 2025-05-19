#!/usr/bin/env python3
"""
CLI entry-point for the AI-Driven Cyber-Threat-Intelligence system.

Run individual stages, or the full pipeline:

    python run.py --component scrape
    python run.py --component preprocess
    python run.py --component categorize
    python run.py --component urgency
    python run.py --component detect_anomalies
    python run.py --component all --dashboard
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import argparse

from dotenv import load_dotenv
from scraper.nvd_scraper import NVDScraper
from scraper.hackernews_scraper import HackerNewsScraper
from preprocessing.clean_text import main as preprocess_main
import pipeline.threat_classifier as threat_classifier
from pipeline.urgency_scoring import main as urgency_main
from pipeline.anomaly_detection import main as anomaly_main
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
LOGS_DIR = ROOT / "logs"


def run_component(name: str, args) -> bool:
    log = logging.getLogger("run")
    log.info(f" Running component: {name}")

    if name == "scrape":
        if args.source in ("all", "hackernews"):
            log.info("  HackerNews")
            out = ROOT / "data/raw/hackernews/hackernews.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not HackerNewsScraper(history_file=str(out)).run():
                log.error("   HackerNews failed")
                return False
        if args.source in ("all", "nvd"):
            log.info("   NVD")
            out = ROOT / "data/raw/nvd/nvd.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not NVDScraper(start_year=2019, history_file=str(out)).run():
                log.error("     NVD failed")
                return False

    elif name == "preprocess":
        log.info("   Preprocessing & linking")
        try:
            preprocess_main()
        except Exception:
            log.exception("     Preprocessing failed")
            return False

    elif name == "categorize":
        log.info("  • Threat categorization")
        try:
            threat_classifier.main()
        except Exception:
            log.exception("     Categorization failed")
            return False

    elif name == "urgency":
        log.info("   Urgency scoring")
        try:
            urgency_main()
        except Exception:
            log.exception("     Urgency assessment failed")
            return False

    elif name == "detect_anomalies":
        log.info("   Emerging-threat detection")
        try:
            ed = anomaly_main()
        except Exception:
            log.exception("    Emerging threat detection failed")
            return False

    elif name == "dashboard":
        dash = ROOT / "dashboard/app.py"
        if not dash.exists():
            log.error("     Dashboard not found")
            return False
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dash)])
        log.info("     Dashboard running at http://localhost:8501")

    else:
        log.error(f"Unknown component '{name}'")
        return False

    log.info(f" {name} completed")
    return True


def run_all(args) -> bool:
    for comp in ("scrape", "preprocess", "categorize", "urgency", "detect_anomalies"):
        if not run_component(comp, args):
            logging.getLogger("run").error(f"Pipeline halted at '{comp}'")
            return False
    if args.dashboard:
        run_component("dashboard", args)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Cyber-Threat-Intelligence CLI")
    p.add_argument(
        "--component",
        choices=["scrape","preprocess","categorize","urgency","detect_anomalies","dashboard","all"],
        default="all"
    )
    p.add_argument("--source", choices=["hackernews","nvd","all"], default="all")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    args = p.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)
    logfile = LOGS_DIR / f"cti_{datetime.utcnow():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    load_dotenv()
    ok = run_all(args) if args.component == "all" else run_component(args.component, args)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
