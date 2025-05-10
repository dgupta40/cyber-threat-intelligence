import requests
import os
import json
from datetime import datetime

# Set up logging to console
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Get API key from environment
api_key = os.getenv('NVD_API_KEY')
logger.info(f"API key present: {api_key is not None}")

# Base URL from documentation
api_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Headers
headers = {}
if api_key:
    headers['apiKey'] = api_key

# Test 1: Basic connection with minimal parameters
logger.info("Test 1: Basic connection")
try:
    response = requests.get(api_url, headers=headers)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.text[:200]}...")
except Exception as e:
    logger.error(f"Error: {e}")

# Test 2: Try a specific CVE ID
logger.info("\nTest 2: Specific CVE")
try:
    params = {"cveId": "CVE-2021-44228"}
    response = requests.get(api_url, headers=headers, params=params)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.text[:200]}...")
except Exception as e:
    logger.error(f"Error: {e}")

# Test 3: Try with keyword search
logger.info("\nTest 3: Keyword search")
try:
    params = {"keywordSearch": "log4j"}
    response = requests.get(api_url, headers=headers, params=params)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.text[:200]}...")
except Exception as e:
    logger.error(f"Error: {e}")

# Try different date format variations
date_formats = [
    {"pubStartDate": "2023-01-01T00:00:00.000", "pubEndDate": "2023-01-31T23:59:59.999"},
    {"pubStartDate": "2023-01-01T00:00:00Z", "pubEndDate": "2023-01-31T23:59:59Z"},
    {"pubStartDate": "2023-01-01", "pubEndDate": "2023-01-31"}
]

for i, date_format in enumerate(date_formats):
    logger.info(f"\nTest {i+4}: Date format: {date_format}")
    try:
        response = requests.get(api_url, headers=headers, params=date_format)
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {response.text[:200]}...")
    except Exception as e:
        logger.error(f"Error: {e}")