#!/usr/bin/env python3
"""
CLI entry-point for the AI-Driven Cyber-Threat-Intelligence system.

Run individual stages, or the full pipeline:

    python run.py --component scrape
    python run.py --component train  --model lgbm
    python run.py --component all    --model all   --dashboard
"""

from __future__ import annotations
import argparse, importlib, json, logging, os, sys, subprocess
from datetime import datetime
from pathlib import Path

# ────────────────────────────── bootstrap ───────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from scraper.hackernews_scraper import HackerNewsScraper
from scraper.nvd_scraper        import NVDScraper
from preprocessing.clean_text   import TextPreprocessor
from anomaly_detection.detect_anomalies import AnomalyDetector
from utils.helpers              import load_env

# ─────────────────────────── pipeline pieces ────────────────────────────
def run_component(component_name: str, args) -> bool:
    """Run one pipeline component; return True on success."""
    log = logging.getLogger("run")
    log.info("Running component: %s", component_name)

    # ──────────────── SCRAPE ────────────────────────────────────────────
    if component_name == "scrape":

        # ── HackerNews ────────────────────────────────────────────────
        if args.source in ("all", "hackernews"):
            log.info("Scraping HackerNews")
            master = ROOT / "data/raw/hackernews/hackernews.json"
            master.parent.mkdir(parents=True, exist_ok=True)

            existing = []
            if master.exists():
                with master.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            seen_urls = {a["url"] for a in existing}

            temp = master.with_name("hackernews_new.json")
            spider = ROOT / "scraper/hackernews_scraper.py"
            cmd = [sys.executable, "-m", "scrapy", "runspider",
                   str(spider), "-a", f"output={temp}"]
            log.debug("Launching spider: %s", " ".join(cmd))
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                log.error("HackerNews spider failed"); return False

            scraped = []
            if temp.exists():
                with temp.open("r", encoding="utf‑8") as fh:
                    scraped = json.load(fh)
                temp.unlink(missing_ok=True)

            new_items = [i for i in scraped if i["url"] not in seen_urls]
            if new_items:
                with master.open("w", encoding="utf‑8") as fh:
                    json.dump(existing + new_items, fh,
                              indent=2, ensure_ascii=False)
                log.info("Added %d new HackerNews articles.", len(new_items))
            else:
                log.info("No new HackerNews articles found.")

        # ── NVD ───────────────────────────────────────────────────────
        if args.source in ("all", "nvd"):
            log.info("Scraping NVD")
            nvd_hist = ROOT / "data/raw/nvd/nvd.json"
            nvd_hist.parent.mkdir(parents=True, exist_ok=True)
            if not NVDScraper(start_year=2019, history_file=nvd_hist).run():
                log.error("NVD scrape failed"); return False
            log.info("NVD up-to-date")

    # ─────────────── PREPROCESS ─────────────────────────────────────────
    elif component_name == "preprocess":
        TextPreprocessor().process_all_sources()
        log.info("Pre-processing completed")

    # ───────────────── TRAIN ───────────────────────────────────────────
    elif component_name == "train":
        trainers = {
            "lr"  : "classification.train_severity_lr",
            "lgbm": "classification.train_severity_lgbm",
        }
        selected = trainers if args.model == "all" else {args.model: trainers[args.model]}
        for key, module_path in selected.items():
            log.info("Training model [%s]", key)
            module = importlib.import_module(module_path)
            if not hasattr(module, "train"):
                log.error("%s has no train() function", module_path); return False
            module.train()
        log.info("Model training completed")

    # ───────────── ANOMALY DETECTION ────────────────────────────────────
    elif component_name == "detect_anomalies":
        AnomalyDetector().detect()
        log.info("Anomaly detection completed")

    # ───────────────── DASHBOARD ────────────────────────────────────────
    elif component_name == "dashboard":
        dash = ROOT / "dashboard/app.py"
        if not dash.exists():
            log.error("dashboard/app.py not found"); return False
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dash)])
        log.info("Dashboard started → http://localhost:8501")

    else:
        log.error("Unknown component: %s", component_name); return False

    return True

# ───────────────────── orchestration helpers ────────────────────────────
def run_all_components(args) -> bool:
    """scrape → preprocess → train → detect_anomalies (optional dash)."""
    log = logging.getLogger("run")
    log.info("Running full pipeline")

    for comp in ("scrape", "preprocess", "train", "detect_anomalies"):
        if not run_component(comp, args):
            log.error("Pipeline halted at '%s'", comp)
            return False

    if args.dashboard:
        run_component("dashboard", args)

    log.info("Pipeline finished successfully")
    return True

# ─────────────────────────────── CLI ────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Cyber‑Threat‑Intelligence CLI")
    p.add_argument("--component",
                   choices=["scrape", "preprocess", "train",
                            "detect_anomalies", "dashboard", "all"],
                   default="all")
    p.add_argument("--source",
                   choices=["hackernews", "nvd", "all"], default="all")
    p.add_argument("--method",
                   choices=["scrapy", "bs4"], default="scrapy")
    p.add_argument("--model",
                   choices=["lr", "lgbm", "all"], default="all")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--log-level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = p.parse_args()

    # logging
    logs_dir = ROOT / "logs"; logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"cti_{datetime.utcnow():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    load_env()
    ok = run_all_components(args) if args.component == "all" \
         else run_component(args.component, args)
    return 0 if ok else 1

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
