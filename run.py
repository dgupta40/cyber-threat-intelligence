#!/usr/bin/env python3
"""
Main execution script for the AI-Driven Cyber Threat Intelligence System.
This script provides a command-line interface to run the complete pipeline
or individual components.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
import json  # Import json for scraping logic

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from scraper.hackernews_scraper import HackerNewsScraper
from scraper.nvd_scraper import NVDScraper
from preprocessing.clean_text import TextPreprocessor
from classification.train_model import ThreatClassifier
from sentiment_analysis.assess_risk import RiskAssessor
from anomaly_detection.detect_anomalies import AnomalyDetector
from utils.helpers import setup_logging, load_env

def run_component(component_name, args):
    """Run a specific component."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running component: {component_name}")

    if component_name == "scrape":
        logger.info("Starting scraping...")
        if args.source == "all" or args.source == "hackernews":
            logger.info("Scraping HackerNews...")
            scraper = HackerNewsScraper(method=args.method)
            master = os.path.join(scraper.output_dir, 'hackernews.json')
            temp   = os.path.join(scraper.output_dir, 'hackernews_new.json')

            # load existing
            if os.path.exists(master):
                with open(master, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            else:
                existing = []

            seen_urls = {item['url'] for item in existing}

            # scrape into temp file
            all_items = scraper.scrape_new(temp)

            # filter out duplicates
            new_items = [i for i in all_items if i['url'] not in seen_urls]

            if new_items:
                combined = existing + new_items
                with open(master, 'w', encoding='utf-8') as f:
                    json.dump(combined, f, indent=2, ensure_ascii=False)
                logger.info(f" Added {len(new_items)} new HackerNews articles.")
            else:
                logger.info(" No new HackerNews articles found.")

            # cleanup
            try:
                os.remove(temp)
            except OSError:
                pass
            logger.info("HackerNews scraping completed")

        if args.source == "all" or args.source == "nvd":
            logger.info("Scraping NVD...")
            nvd_output = os.path.join('data','raw','nvd','nvd.json')
            os.makedirs(os.path.dirname(nvd_output), exist_ok=True)
            nvd_scraper = NVDScraper(start_year=2019, history_file=nvd_output) # Using the logic from main.py
            if nvd_scraper.run():
                logger.info(f" NVD: up‑to‑date (history in {nvd_output})")
            else:
                logger.error(" NVD: scrape failed")
            logger.info("NVD scraping completed")

    elif component_name == "preprocess":
        # Preprocess data
        preprocessor = TextPreprocessor()
        preprocessor.process_all_sources()
        logger.info("Preprocessing completed")

    elif component_name == "train":
        # Train classification models
        classifier = ThreatClassifier()
        classifier.train()
        logger.info("Model training completed")

    elif component_name == "analyze_risk":
        # Analyze risk
        risk_assessor = RiskAssessor()
        risk_assessor.analyze()
        logger.info("Risk assessment completed")

    elif component_name == "detect_anomalies":
        # Detect anomalies
        anomaly_detector = AnomalyDetector()
        anomaly_detector.detect()
        logger.info("Anomaly detection completed")

    elif component_name == "evaluate":
        # Run evaluation script
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_evaluation.py")
        if os.path.exists(eval_script):
            subprocess.run([sys.executable, eval_script])
            logger.info("Model evaluation completed")
        else:
            logger.error(f"Evaluation script not found: {eval_script}")

    elif component_name == "dashboard":
        # Run Streamlit dashboard
        dashboard_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard/app.py")
        if os.path.exists(dashboard_script):
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_script])
            logger.info(f"Dashboard started: http://localhost:8501")
        else:
            logger.error(f"Dashboard script not found: {dashboard_script}")

    else:
        logger.error(f"Unknown component: {component_name}")
        return False

    return True

def run_all_components(args):
    """Run all components in sequence."""
    logger = logging.getLogger(__name__)
    logger.info("Running complete pipeline")

    # Create required directories
    for directory in ["data/raw/hackernews", "data/raw/nvd", "data/processed", "models", "logs", "evaluations"]:
        os.makedirs(directory, exist_ok=True)

    # Run components in sequence
    components = ["scrape", "preprocess", "train", "analyze_risk", "detect_anomalies", "evaluate"]

    for component in components:
        logger.info(f"Starting component: {component}")
        success = run_component(component, args)

        if not success:
            logger.error(f"Pipeline failed at component: {component}")
            return False

    # Start dashboard if requested
    if args.dashboard:
        run_component("dashboard", args)

    logger.info("Complete pipeline execution finished successfully")
    return True

def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run AI-Driven Cyber Threat Intelligence System")

    # Add arguments
    parser.add_argument("--component", choices=["scrape", "preprocess", "train", "analyze_risk", "detect_anomalies", "evaluate", "dashboard", "all"],
                        default="all", help="Component to run (default: all)")

    parser.add_argument("--source", choices=["hackernews", "nvd", "all"],
                        default="all", help="Data source to scrape (default: all)")

    parser.add_argument("--method", choices=["scrapy", "bs4"],
                        default="scrapy", help="Scraping method for HackerNews (default: scrapy)")

    parser.add_argument("--dashboard", action="store_true",
                        help="Start the dashboard after pipeline completion")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level (default: INFO)")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"cti_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


    logger = logging.getLogger(__name__)
    logger.info(f"Starting Cyber Threat Intelligence System - Component: {args.component}")

    # Load environment variables
    load_env()

    # Run pipeline
    if args.component == "all":
        success = run_all_components(args)
    else:
        success = run_component(args.component, args)

    if success:
        logger.info("Execution completed successfully")
        return 0
    else:
        logger.error("Execution failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())