import logging
from pathlib import Path
from . import Scraper
import lxml.html
import requests
import csv
import time
from typing import Dict, Optional
from utils import get_child_logger, retry_on_exception, remove_citations, load_json, save_json


class TNCZScraper(Scraper):
    def __init__(self,
                 run_dir: Path,
                 output_dir: Optional[Path] = None,
                 user_agent: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        output_dir = output_dir or Path('scraped_data/tn_cz')
        output_dir = output_dir.resolve()
        super().__init__(output_dir, user_agent, get_child_logger(__name__, logger))

        self.run_dir = run_dir.resolve()
        self.users: Dict[int, bool] = {}
        self.output_file: Optional[Path] = None
        self.sleep_time = 0.2
        self.session = requests.Session()
        self._update_session_headers()

    def _update_session_headers(self):
        self.session.headers.update({'User-Agent': self.user_agent})

    def load_state(self):
        state = load_json(self.run_dir / 'last_state.json')
        self.output_dir = Path(state['output_dir'])
        self.output_file = Path(state['output_file'])
        self.user_agent = state['user_agent']
        self.users = {int(user_id): scraped for user_id, scraped in state['users'].items()}
        self._update_session_headers()
        self.logger.info("Resuming scraping from the last saved state. "
                         f"Data will be appended to '{self.output_file}'")

    def save_state(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'output_dir': str(self.output_dir),
            'output_file': str(self.output_file),
            'user_agent': self.user_agent,
            'users': self.users
        }
        save_json(state, self.run_dir / 'last_state.json')

    @retry_on_exception(requests.exceptions.RequestException)  # noqa
    def _get_html(self, url: str):
        response = self.session.get(url)
        response.raise_for_status()
        return lxml.html.fromstring(response.text)

    def _get_page(self, url: str) -> Optional[lxml.html.HtmlElement]:
        try:
            html = self._get_html(url)
        except RuntimeError as exc:
            if isinstance(exc.__cause__, requests.exceptions.HTTPError) and exc.__cause__.response.status_code == 404:  # noqa
                return None
            if isinstance(exc.__cause__, requests.exceptions.MissingSchema):
                return None
            raise
        return html

    @classmethod
    def _parse_article(cls, text: str) -> Optional[str]:
        text = remove_citations(text)
        return cls._parse_author_text(text, check_language=False)

    def _scrape_article(self, user_id: int, article_url: str, writer: csv.writer) -> bool:
        html = self._get_page(article_url)
        if html is None:
            return False

        ignore_cls = ['c-card', 'c-player', 'twitter-tweet', 'instagram-media', 'c-inline-gallery', 'img']
        for cls in ignore_cls:
            elements_to_del = html.xpath(f'//*[contains(@class, "{cls}")]')
            for element in elements_to_del:
                element.getparent().remove(element)

        p_list = html.xpath('//div[contains(@class, "c-hero-wrapper")]//div[contains(@class, "c-content-inner")]//div[@class="c-rte"]''//p')  # noqa
        text = '\n'.join([p.text_content().strip() for p in p_list if p.text_content().strip()])
        text = self._parse_article(text)
        if text:
            writer.writerow((user_id, article_url, text))
            return True
        return False

    def _scrape_user(self, user_id: int, writer: csv.writer, limit: Optional[int] = None):
        page = 1
        articles = 0
        html = self._get_page(f'https://tn.nova.cz/autor/{user_id}/strana-{page}')
        if html is None or html.xpath('//div[@class="c-404"]'):
            self.logger.info(f"User {user_id} not found, skipping")
            return

        try:
            page_nav = html.xpath(
                '//div[contains(@class, "c-content-inner")]//section[@class="c-section-inner"]//nav[@class="c-pagination"]')[0]  # noqa
            last_page = int(page_nav.xpath('.//li/a')[-2].text)
        except IndexError:
            last_page = 1

        while not _limit_reached(articles, limit) and page <= last_page:
            if page > 1:
                html = self._get_page(f'https://tn.nova.cz/autor/{user_id}/strana-{page}')
                if html is None:
                    break
            for section in html.xpath(
                    '//div[contains(@class, "c-content-inner")]//section[@class="c-section-inner"]'):  # noqa
                for article_url in section.xpath(
                        './/article[contains(@class, "c-article")]//h3[@class="title"]/a/@href'):
                    if self._scrape_article(user_id, article_url, writer):
                        articles += 1
            self.logger.debug(f"Scraped page {page}/{last_page} of user {user_id}")
            page += 1
        self.logger.info(f"Scraped all articles of user {user_id}")

    def scrape(self, limit: Optional[int] = None):
        self._init_directories()
        self.users = self.users or {user_id: False for user_id in range(1, 1000)}
        self.output_file = self.output_file or self.output_dir / f'tncz_{time.strftime("%y%m%d_%H%M%S")}.csv'  # noqa
        self.save_state()

        users_to_scrape = [user_id for user_id, scraped in self.users.items() if not scraped]
        if not users_to_scrape:
            self.logger.info('No users to scrape')
            return

        with self.output_file.open('a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('author', 'article_url', 'text'))  # Header
            # Enumerate all users
            for user_id in users_to_scrape:
                self._scrape_user(user_id, writer, limit)
                self.users[user_id] = True
                self.save_state()

def _limit_reached(current: int, maximum: Optional[int]) -> bool:
    return maximum is not None and current >= maximum
