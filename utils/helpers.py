"""
Helper functions for the AI-Driven Cyber Threat Intelligence System.
"""

import os
import logging
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
def load_env():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set TensorFlow log level from environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')
    
    return True

def setup_directories(directories):
    """
    Create necessary directories if they don't exist.
    
    Args:
        directories (list): List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_logging():
    """Configure logging for the application."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"cyber_threat_intel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_to_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename (str): Path to save the file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_from_json(filename):
    """
    Load data from a JSON file.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filename}")
        return {}


def save_to_csv(data, filename, headers=None):
    """
    Save data to a CSV file.
    
    Args:
        data (list): List of dictionaries or list of lists
        filename (str): Path to save the file
        headers (list, optional): List of column headers
    """
    mode = 'w'
    file_exists = os.path.isfile(filename)
    
    if isinstance(data[0], dict) and headers is None:
        headers = list(data[0].keys())
    
    with open(filename, mode, newline='', encoding='utf-8') as f:
        if isinstance(data[0], dict):
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(f)
            if headers and not file_exists:
                writer.writerow(headers)
            writer.writerows(data)


def load_from_csv(filename, as_dict=True):
    """
    Load data from a CSV file.
    
    Args:
        filename (str): Path to the file
        as_dict (bool): Return data as list of dictionaries if True,
                       list of lists if False
                       
    Returns:
        list: Loaded data
    """
    try:
        if as_dict:
            return pd.read_csv(filename).to_dict('records')
        else:
            return pd.read_csv(filename).values.tolist()
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return []
    except Exception as e:
        logging.error(f"Error loading CSV file {filename}: {str(e)}")
        return []


def get_all_files(directory, extension=None):
    """
    Get all files in a directory, optionally filtered by extension.
    
    Args:
        directory (str): Directory path
        extension (str, optional): File extension to filter by (e.g., '.json')
        
    Returns:
        list: List of file paths
    """
    files = []
    for file in os.listdir(directory):
        if extension is None or file.endswith(extension):
            files.append(os.path.join(directory, file))
    return files


def merge_dataframes(dataframes, on=None):
    """
    Merge multiple dataframes.
    
    Args:
        dataframes (list): List of pandas DataFrames
        on (str or list, optional): Column(s) to join on
        
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    result = dataframes[0]
    for df in dataframes[1:]:
        if on is not None:
            result = pd.merge(result, df, on=on, how='outer')
        else:
            result = pd.concat([result, df], ignore_index=True)
    
    return result


def get_env_variable(key, default=None):
    """
    Get environment variable with a default value.
    
    Args:
        key (str): Environment variable name
        default: Default value if not found
        
    Returns:
        str: Environment variable value
    """
    return os.environ.get(key, default)


def format_timestamp(timestamp=None):
    """
    Format timestamp string for consistent usage.
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        
    Returns:
        str: Formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return timestamp
                
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(text):
    """
    Basic text normalization.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def chunk_list(lst, chunk_size):
    """
    Split a list into chunks of specified size.
    
    Args:
        lst (list): List to chunk
        chunk_size (int): Size of each chunk
        
    Returns:
        list: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_request(url, max_retries=3, timeout=10, **kwargs):
    """
    Make a safe HTTP request with retries.
    
    Args:
        url (str): URL to request
        max_retries (int): Maximum number of retries
        timeout (int): Request timeout in seconds
        **kwargs: Additional arguments to pass to requests.get
        
    Returns:
        requests.Response or None: Response object or None on failure
    """
    import requests
    from requests.exceptions import RequestException
    import time
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except RequestException as e:
            logging.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Request failed after {max_retries} attempts: {url}")
                return None