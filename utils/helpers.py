#!/usr/bin/env python3
"""
helpers.py
Core JSON I/O helpers for CTI pipeline
"""

import os
import json
import gzip
import logging

import ijson


def load_from_json(path: str) -> any:
    """
    Load JSON or GZIP JSON files; stream if larger than 200MB.

    Returns parsed JSON object or empty dict on failure.
    """
    opener = gzip.open if path.endswith((".gz", ".gzip")) else open
    try:
        if os.path.getsize(path) > 200 * 1024 * 1024:
            with opener(path, "rb") as fh:
                return list(ijson.items(fh, ""))
        with opener(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning(f"Error loading JSON file {path}: {e}")
        return {}


def save_to_json(obj: any, path: str, **json_kwargs) -> None:
    """
    Save Python object to JSON file, ensuring directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, **json_kwargs)


__all__ = [
    "load_from_json",
    "save_to_json",
]
