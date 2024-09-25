import csv
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional

import lxml.html
import requests

from utils import get_child_logger, load_json, retry_on_exception, save_json
from . import Scraper

MAX_REVIEWS_PER_USER = 6000


class CSFDScraper(Scraper):
    def __init__(self,
                 run_dir: Path,
                 output_dir: Optional[Path] = None,
                 user_agent: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        output_dir = output_dir or Path('scraped_data/csfd')
        output_dir = output_dir.resolve()
        super().__init__(output_dir, user_agent, get_child_logger(__name__, logger))

        self.run_dir = run_dir.resolve()
        self.users: Dict[str, bool] = {}
        self.output_file: Optional[Path] = None
        self.sleep_time = 0.2
        self.session = requests.Session()
        self._update_session_headers()

    def _update_session_headers(self):
        self.session.headers.update({'User-Agent': self.user_agent})

    @retry_on_exception(requests.exceptions.RequestException)  # noqa
    def _get_html(self, url: str):
        response = self.session.get(url)
        response.raise_for_status()
        return lxml.html.fromstring(response.text)

    def _scrape_users(self) -> Dict[str, bool]:
        html = self._get_html('https://www.csfd.cz/uzivatele/nejaktivnejsi/?country=1')
        user_urls = html.xpath(
            '//h2[text()="Recenze"]/../following-sibling::div//div[@class="article-content"]//a[@class="user-title-name"]//@href')  # noqa
        return {self._parse_username_from_url(url): False for url in user_urls}

    def _scrape_reviews(self, username: str, writer: csv.writer):
        page = 1
        reviews = 0

        while reviews < MAX_REVIEWS_PER_USER:
            self.logger.debug(f"Scraping page #{page} of user '{username}'")
            html = self._get_html(f'https://www.csfd.cz/uzivatel/{username}/recenze/?page={page}')

            for review in html.xpath(
                    '//section[@class="box striped-articles user-reviews"]/div[@class="box-content"]/article'):  # noqa
                movie_url = review.xpath('.//a[@class="film-title-name"]/@href')[0]
                movie_id = self._parse_movie_id_from_url(movie_url)
                review_text = ''.join(
                    review.xpath('.//div[@class="user-reviews-text"]//span[@class="comment"]')[
                        0].itertext())  # noqa
                review_text = self._parse_author_text(review_text, True)
                if review_text:
                    writer.writerow((username, movie_id, review_text))
                    reviews += 1

            if not html.xpath('//div[@class="user-main-content"]//a[@class="page-next"]'):
                self.logger.debug(f"Scraped all reviews of user '{username}'")
                return

            page += 1
            time.sleep(self.sleep_time)
        self.logger.debug(f"Scraped {reviews} reviews of user '{username}' (limit reached)")

    def scrape(self):
        self._init_directories()
        self.users = self.users or self._scrape_users()
        self.output_file = self.output_file or self.output_dir / f'csfd_{time.strftime("%y%m%d_%H%M%S")}.csv'  # noqa
        self.save_state()

        users_to_scrape = [user for user, scraped in self.users.items() if not scraped]
        user_count = len(users_to_scrape)

        if user_count == 0:
            self.logger.info('No users to scrape')
            return

        self.logger.info(f"Scraping started. Output file: '{self.output_file}'")
        start_time = time.time()
        with self.output_file.open('a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('author', 'movie_id', 'text'))  # Header
            for i, user in enumerate(users_to_scrape):
                self._scrape_reviews(user, writer)
                self.users[user] = True
                self.save_state()
                self.logger.info(f"Finished scraping of user '{user}' ({i + 1}/{user_count})")
        time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        self.logger.info(f"Scraping finished. Time elapsed: {time_elapsed}")

    def load_state(self):
        state = load_json(self.run_dir / 'last_state.json')
        self.output_dir = Path(state['output_dir'])
        self.output_file = Path(state['output_file'])
        self.user_agent = state['user_agent']
        self.users = state['users']
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

    def _parse_movie_id_from_url(self, url: str) -> str:
        pattern = r"film/(\d+)(?:-[^/]+)?(?:/(\d+))?"
        matches = re.search(pattern, url)
        if matches:
            movie_id = matches.group(1)
            episode_id = matches.group(2)
            return f"{movie_id}_{episode_id}" if episode_id else movie_id
        self.logger.warning(f"Failed to parse movie ID from URL '{url}'")
        return 'UNKNOWN'

    @classmethod
    def _parse_username_from_url(cls, url: str) -> str:
        return url.split('uzivatel/', maxsplit=1)[-1].split('/', maxsplit=1)[0]

    @classmethod
    def _parse_review_count(cls, text: str) -> int:
        for space in (' ', '\xa0'):
            text = text.replace(space, '')
        return int(text.split('rec', maxsplit=1)[0])
