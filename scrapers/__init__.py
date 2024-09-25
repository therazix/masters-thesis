import logging
from pathlib import Path
from typing import Optional

import langid

import utils

DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'  # noqa
MIN_AUTHOR_TEXT_LENGTH = 20


class Scraper:
    def __init__(self, output_dir: Path, user_agent: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.output_dir = output_dir
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.logger = utils.get_child_logger(__name__, logger)

    def _init_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape(self):
        raise NotImplementedError

    @classmethod
    def _parse_author_text(cls, text: str, check_language=True) -> Optional[str]:
        text = text.strip()  # Remove leading and trailing whitespaces
        if not text or text.isspace():
            return None
        if len(text) < MIN_AUTHOR_TEXT_LENGTH:
            return None
        if check_language and not cls._is_correct_lang(text):
            return None
        return text

    @classmethod
    def _is_correct_lang(cls, text: str) -> bool:
        lang, _ = langid.classify(text)
        # Text is sometime wrongly classified as 'sk', so we also accept 'sk' language
        return lang == 'cs' or lang == 'sk'
