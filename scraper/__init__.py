import logging
import pycld2 as cld2
from typing import Optional
from pathlib import Path

DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'  # noqa
MIN_AUTHOR_TEXT_LENGTH = 20
LOGGER = logging.getLogger(__name__)


class Scraper:
    def __init__(self, output_dir: Path, user_agent: Optional[str]):  # noqa
        self.output_dir = output_dir
        self.user_agent = user_agent or DEFAULT_USER_AGENT

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
        if check_language and not cls._is_czech_text(text):
            return None
        return text

    @classmethod
    def _is_czech_text(cls, text: str) -> bool:
        _, _, details = cld2.detect(text, bestEffort=True, hintLanguage='cs')
        return details[0][1] == 'cs'
