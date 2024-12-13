import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import strip_markdown

MIN_AUTHOR_TEXT_LENGTH = 8
LOGGER = logging.getLogger(__name__)


def retry_on_exception(exception: BaseException, max_retries: int = 5, sleep_time: int = 5):  # noqa
    def decorator(func):
        def _retry(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exception as exc:
                    last_exc = exc
                    time.sleep(sleep_time)
                    LOGGER.warning(f"Function '{func.__name__}' raised an exception: '{exc}'. "
                                   f"Retry {i + 1}/{max_retries}.")
            raise RuntimeError(
                f"Failed to execute '{func.__name__}', max retries exceeded.") from last_exc
        return _retry
    return decorator


def clean_text(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = re.sub(r'http\S+|www\.\S+|https\S+', 'URL', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def remove_markdown(text: str) -> str:
    # Create a pattern to match lines starting with '>' (blockquotes),
    # or lines that starts with '^(' and ends with ')' (footnotes)
    pattern = re.compile(r'(^>.*$\n?|^\^\(.*\)$\n?)', re.MULTILINE)
    text = pattern.sub('', text)
    return strip_markdown.strip_markdown(text)


def remove_citations(text: str) -> str:
    text = re.sub(r'["„‚][^"„‚“‘]+,["“‘].*?\.', '', text)
    return re.sub(r'["„‚][^"„‚“‘]+[.]["“‘]', '', text)


def load_json(file_path: Path):
    with file_path.open('r') as file:
        return json.load(file)


def save_json(data, file_path: Path):
    with file_path.open('w') as file:
        json.dump(data, file, indent=2)


def load_csv(file_path: Path):
    return pd.read_csv(file_path)


def get_child_logger(name: str, parent_logger: Optional[logging.Logger] = None) -> logging.Logger:
    if parent_logger:
        return parent_logger.getChild(name)
    else:
        return logging.getLogger(name)
