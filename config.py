"""
Configuration settings for the AI-Driven Cyber Threat Intelligence System.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EVALUATIONS_DIR = os.path.join(BASE_DIR, 'evaluations')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, EVALUATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Scraper configurations
SCRAPER_CONFIG = {
    'hackernews': {
        'url': 'https://thehackernews.com/search/label/Vulnerability',
        'max_pages': 5,
        'user_agent': 'Mozilla/5.0 (compatible; CTIBot/1.0)',
        'output_dir': os.path.join(RAW_DIR, 'hackernews')
    },
    'nvd': {
        'start_year': 2019,
        'api_key': os.getenv('NVD_API_KEY', None),
        'api_url': 'https://services.nvd.nist.gov/rest/json/cves/2.0',
        'output_dir': os.path.join(RAW_DIR, 'nvd')
    }
}

# Model configurations
MODEL_CONFIG = {
    'classification': {
        'traditional': {
            'vectorizer': {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.85
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000
            },
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42
            },
            'svm': {
                'C': 1.0
            },
            'naive_bayes': {
                'alpha': 0.1
            }
        },
        'deep_learning': {
            'embedding_dim': 128,
            'max_len': 200,
            'batch_size': 32,
            'epochs': 10
        },
        'transformer': {
            'model_name': 'distilbert-base-uncased',
            'max_length': 128,
            'batch_size': 16,
            'epochs': 3
        }
    },
    'risk_assessment': {
        'vader_thresholds': {
            'high_risk': -0.25,
            'medium_risk': 0.25
        },
        'combined_thresholds': {
            'high_risk': 0.7,
            'medium_risk': 0.3
        },
        'cvss_weights': {
            'sentiment': 0.3,
            'cvss': 0.7
        }
    },
    'anomaly_detection': {
        'isolation_forest': {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': 42
        },
        'local_outlier_factor': {
            'n_neighbors': 20,
            'contamination': 0.1
        },
        'autoencoder': {
            'hidden_dim': 128,
            'latent_dim': 32,
            'learning_rate': 1e-3,
            'epochs': 50
        }
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'title': 'AI-Driven Cyber Threat Intelligence Dashboard',
    'port': 8501,
    'theme': 'light',
    'default_tabs': [
        'Overview',
        'Threat Categories',
        'Risk Assessment',
        'Anomaly Detection',
        'Detailed View'
    ]
}

# Logging configuration
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Threat categories
THREAT_CATEGORIES = [
    'Phishing',
    'Ransomware',
    'Malware',
    'SQLInjection',
    'XSS',
    'DDoS',
    'ZeroDay',
    'SupplyChain',
    'DataBreach',
    'APT',
    'Other'
]

# Risk levels
RISK_LEVELS = ['Low', 'Medium', 'High']

# Scheduler configuration
SCHEDULER_CONFIG = {
    'scrape_interval': int(os.getenv('SCRAPE_INTERVAL_HOURS', 6)),
    'process_interval': int(os.getenv('PROCESS_INTERVAL_HOURS', 12)),
    'train_interval': int(os.getenv('TRAIN_INTERVAL_DAYS', 7)) * 24  # Convert to hours
}