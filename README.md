# AI-Driven Cyber Threat Intelligence

A modular pipeline and dashboard for collecting, processing, classifying, and analyzing cyber vulnerability data from NVD and The Hacker News.

---

## 🚀 Project Overview

This project automates the ingestion, processing, and analysis of cybersecurity vulnerability data:

1. **Scraping**

   * Fetch CVE data from [NVD API](https://nvd.nist.gov) (incremental updates)
   * Scrape vulnerability articles from [The Hacker News](https://thehackernews.com)

2. **Preprocessing & Linking** (`clean_text.py`)

   * HTML → text cleaning
   * Normalize and mask PII/CVE references
   * Extract CVSS bins, perform sentiment analysis, TF‑IDF, and SBERT embeddings
   * Link articles ↔ CVEs (count, earliest date)

3. **Threat Classification** (`threat_classifier.py`)

   * Multi-label classification into categories (Phishing, Malware, XSS, etc.)

4. **Urgency Scoring** (`urgency_scoring.py`)

   * Compute a composite urgency score from severity, sentiment, recency, and exploit/patch indicators
   * Discretize into Low/Medium/High levels

5. **Emerging Threat Detection** (`anomaly_detection.py`)

   * Zero-day keyword heuristic
   * Spike detection on mention counts
   * Isolation Forest on random-projected TF‑IDF vectors

6. **Dashboard** (`dashboard/app.py`) (Work in Progress)

   * Streamlit UI for overview metrics, timelines, category distributions, urgency & emerging threats, and CVE details

7. **Automation**

   * **`scheduler.py`**: run scrapes every 6 hours
   * **`run.py`**: CLI orchestrator for pipeline stages and dashboard

---

## 📦 Prerequisites

* **Python 3.8+**
* **Virtual environment** (recommended)
* **NVD API key** in a `.env` file:

  ```bash
  NVD_API_KEY=YOUR_API_KEY_HERE
  ```

---

## ⚙️ Installation

```bash
git clone https://github.com/dgupta40/cyber-threat-intelligence.git
cd cyber-threat-intelligence
python -m venv .venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run scheduler (periodic scraping)

```bash
python scheduler.py (Still to be implemented in the main pipline)
```

Runs an immediate scrape for NVD & Hacker News, then every 6 hours.

### 2. Full pipeline & dashboard

```bash
# Run entire pipeline and launch dashboard
python run.py --component all --dashboard
```

Or run stages individually:

```bash
python run.py --component scrape       # Scrape data
python run.py --component preprocess  # Clean & link
python run.py --component categorize  # Threat classification
python run.py --component urgency     # Urgency scoring
python run.py --component detect_anomalies  # Emerging threats
python run.py --component dashboard   # Open Streamlit dashboard
```

---

## 🗂️ Directory Structure

```
project-root/
├── data/
│   ├── raw/                # JSON from scrapers
│   └── processed/          # Parquet/CSV outputs
├── dashboard/              # Streamlit app
│   └── app.py
├── scraper/                # Scraper modules
│   ├── hackernews_scraper.py
│   └── nvd_scraper.py
├── preprocessing/          # Cleaning & linking
│   └── clean_text.py
├── pipeline/               # Analysis stages
│   ├── threat_classifier.py
│   ├── urgency_scoring.py
│   └── anomaly_detection.py
├── utils/                  # Shared helpers
│   └── helpers.py
├── scheduler.py            # APScheduler job definitions
├── run.py                  # CLI orchestrator
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -m 'Add xyz'`)
4. Push to branch (`git push origin feature/xyz`)
5. Open a Pull Request