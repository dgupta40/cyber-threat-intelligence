#!/usr/bin/env python3
"""
script for the AI-Driven Cyber Threat Intelligence System.
This script coordinates the execution of the entire pipeline:
1. Data collection (scraping)
2. Data preprocessing
3. Model training
4. Threat classification
5. Risk assessment
6. Anomaly detection
"""

import os
import logging
import argparse
from datetime import datetime

# Import modules for each pipeline stage
from scraper.hackernews_scraper import HackerNewsScraper
from scraper.nvd_scraper import NVDScraper
from preprocessing.clean_text import TextPreprocessor
from classification.train_model import ThreatClassifier
from sentiment_analysis.assess_risk import RiskAssessor
from anomaly_detection.detect_anomalies import AnomalyDetector
from utils.helpers import setup_logging, load_env

def run_pipeline(steps=None):
    """
    Run the complete data pipeline or specific steps.
    
    Args:
        steps (list): List of steps to run, or None for all steps
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Cyber Threat Intelligence pipeline")
    
    # Load environment variables
    load_env()
    
    # Define all possible pipeline steps
    all_steps = [
        "scrape", "preprocess", "train", "classify", "assess_risk", "detect_anomalies"
    ]
    
    # If no steps specified, run all
    if steps is None:
        steps = all_steps
    
    # Create required directories
    os.makedirs("data/raw/hackernews", exist_ok=True)
    os.makedirs("data/raw/nvd", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Step 1: Data Collection (Scraping)
    if "scrape" in steps:
        logger.info("Step 1: Data Collection")
        
        # Scrape HackerNews
        hn_scraper = HackerNewsScraper()
        hn_output = os.path.join("data/raw/hackernews", "hackernews_new.json")
        logger.info("Scraping HackerNews...")
        hn_scraper.scrape_new(hn_output)
        
        # Scrape NVD
        nvd_scraper = NVDScraper()
        logger.info("Scraping NVD...")
        nvd_scraper.run()
    
    # Step 2: Data Preprocessing
    if "preprocess" in steps:
        logger.info("Step 2: Data Preprocessing")
        preprocessor = TextPreprocessor()
        preprocessor.process_all_sources()
    
    # Step 3: Model Training
    if "train" in steps:
        logger.info("Step 3: Model Training")
        classifier = ThreatClassifier()
        classifier.train()
    
    # Step 4: Threat Classification
    if "classify" in steps:
        logger.info("Step 4: Threat Classification")
        # This would apply the trained model to classify new threats
        # Currently integrated with training in ThreatClassifier class
        logger.info("Classification is handled during training")
    
    # Step 5: Risk Assessment
    if "assess_risk" in steps:
        logger.info("Step 5: Risk Assessment")
        risk_assessor = RiskAssessor()
        risk_assessor.analyze()
    
    # Step 6: Anomaly Detection
    if "detect_anomalies" in steps:
        logger.info("Step 6: Anomaly Detection")
        anomaly_detector = AnomalyDetector()
        anomaly_detector.detect()
    
    logger.info("Pipeline execution completed")
    return True


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Cyber Threat Intelligence pipeline")
    parser.add_argument(
        "--steps", 
        nargs="+", 
        choices=["scrape", "preprocess", "train", "classify", "assess_risk", "detect_anomalies", "all"],
        help="Specify which pipeline steps to run"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine which steps to run
    steps_to_run = None  # Default: run all steps
    if args.steps:
        if "all" in args.steps:
            steps_to_run = None  # Run all steps
        else:
            steps_to_run = args.steps
    
    # Run the pipeline
    run_pipeline(steps_to_run)