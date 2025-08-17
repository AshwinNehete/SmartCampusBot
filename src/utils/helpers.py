"""
Utility functions for SmartCampusBot
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any
import time
import functools


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove special characters that might interfere with processing
    text = re.sub(r"[^\w\s\.\,\?\!\-\:\;]", "", text)

    return text


def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text

    Args:
        text: Input text

    Returns:
        List of keywords
    """
    # Simple keyword extraction
    words = re.findall(r"\b\w{3,}\b", text.lower())

    # Filter out common stop words
    stop_words = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "who",
        "boy",
        "did",
        "she",
        "use",
        "her",
        "way",
        "many",
    }

    keywords = [word for word in words if word not in stop_words]

    return list(set(keywords))  # Remove duplicates


def categorize_query(query: str) -> str:
    """
    Categorize user query based on keywords

    Args:
        query: User query

    Returns:
        Category name
    """
    query_lower = query.lower()

    category_keywords = {
        "registration": [
            "register",
            "enroll",
            "class",
            "course",
            "add",
            "drop",
            "schedule",
        ],
        "financial": [
            "tuition",
            "payment",
            "financial",
            "aid",
            "scholarship",
            "fee",
            "money",
            "cost",
        ],
        "academic": [
            "grade",
            "gpa",
            "graduation",
            "degree",
            "transcript",
            "requirement",
            "credit",
        ],
        "facilities": [
            "library",
            "dining",
            "food",
            "building",
            "facility",
            "gym",
            "recreation",
        ],
        "services": [
            "health",
            "counseling",
            "career",
            "service",
            "support",
            "advisor",
            "id card",
        ],
        "technology": [
            "wifi",
            "internet",
            "computer",
            "tech",
            "password",
            "login",
            "system",
        ],
        "transportation": ["parking", "shuttle", "bus", "transport", "car", "bike"],
    }

    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return category

    return "general"


def format_response_time(seconds: float) -> str:
    """
    Format response time for display

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def load_json_file(file_path: str) -> Optional[Dict]:
    """
    Load JSON file with error handling

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str) -> bool:
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving {file_path}: {e}")
        return False


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator to retry function with exponential backoff

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2**attempt)
                        logging.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed")

            raise last_exception

        return wrapper

    return decorator


def validate_config(config_dict: Dict) -> List[str]:
    """
    Validate configuration dictionary

    Args:
        config_dict: Configuration dictionary

    Returns:
        List of validation errors
    """
    errors = []

    required_keys = [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "EMBEDDING_MODEL",
        "VECTOR_STORE_PATH",
    ]

    for key in required_keys:
        if key not in config_dict or not config_dict[key]:
            errors.append(f"Missing or empty required configuration: {key}")

    # Validate URL format
    if "OLLAMA_BASE_URL" in config_dict:
        url = config_dict["OLLAMA_BASE_URL"]
        if not (url.startswith("http://") or url.startswith("https://")):
            errors.append("OLLAMA_BASE_URL must start with http:// or https://")

    # Validate numeric values
    numeric_keys = ["TOP_K_RESULTS", "GRADIO_PORT", "MAX_RESPONSE_LENGTH"]
    for key in numeric_keys:
        if key in config_dict:
            try:
                int(config_dict[key])
            except (ValueError, TypeError):
                errors.append(f"{key} must be a valid integer")

    return errors


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
