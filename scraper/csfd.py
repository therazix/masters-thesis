import requests
import lxml.html
import re
import csv
import time
import logging

from utils import retry_on_exception
from . import Scraper
from pathlib import Path
from typing import List, Optional

MAX_REVIEWS_PER_USER = 6000
LOGGER = logging.getLogger(__name__)


class CSFDScraper(Scraper):
    def __init__(self, output_dir: Optional[Path], user_agent: Optional[str]):
        output_dir = output_dir or Path('scraped_data/csfd')
        output_dir = output_dir.resolve()
        super().__init__(output_dir, user_agent)

        self.sleep_time = 0.2
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    @retry_on_exception(requests.exceptions.RequestException)  # noqa
    def _get_html(self, url: str):
        response = self.session.get(url)
        response.raise_for_status()
        return lxml.html.fromstring(response.text)

    def _get_users(self) -> List[str]:
        html = self._get_html('https://www.csfd.cz/uzivatele/nejaktivnejsi/?country=1')
        user_urls = html.xpath(
            '//h2[text()="Recenze"]/../following-sibling::div//div[@class="article-content"]//a[@class="user-title-name"]//@href')  # noqa
        return [self._parse_username_from_url(url) for url in user_urls]

    def _scrape_reviews(self, username: str, writer: csv.writer):
        page = 1
        reviews = 0

        while reviews < MAX_REVIEWS_PER_USER:
            LOGGER.debug(f"Scraping page #{page} of user '{username}'")
            html = self._get_html(f'https://www.csfd.cz/uzivatel/{username}/recenze/?page={page}')

            for review in html.xpath('//section[@class="box striped-articles user-reviews"]/div[@class="box-content"]/article'):  # noqa
                movie_url = review.xpath('.//a[@class="film-title-name"]/@href')[0]
                movie_id = self._parse_movie_id_from_url(movie_url)
                review_text = ''.join(review.xpath('.//div[@class="user-reviews-text"]//span[@class="comment"]')[0].itertext())  # noqa
                review_text = self._parse_author_text(review_text, True)
                if review_text:
                    writer.writerow((username, movie_id, review_text))
                    reviews += 1

            if not html.xpath('//div[@class="user-main-content"]//a[@class="page-next"]'):
                LOGGER.debug(f"Scraped all reviews of user '{username}'")
                return

            page += 1
            time.sleep(self.sleep_time)
        LOGGER.debug(f"Scraped {reviews} reviews of user '{username}' (limit reached)")

    def scrape(self):
        LOGGER.info('Started scraping CSFD')
        self._init_directories()

        users = self._get_users()
        user_count = len(users)

        filepath = self.output_dir / f'csfd_{time.strftime("%y%m%d_%H%M%S")}.csv'

        with filepath.open('a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('author', 'movie_id', 'text'))  # Header
            for i, user in enumerate(users):
                self._scrape_reviews(user, writer)
                LOGGER.info(f"Finished scraping of user '{user}' ({i + 1}/{user_count})")
        LOGGER.info('Scraping finished')

    @classmethod
    def _parse_username_from_url(cls, url: str) -> str:
        return url.split('uzivatel/', maxsplit=1)[-1].split('/', maxsplit=1)[0]

    @classmethod
    def _parse_movie_id_from_url(cls, url: str) -> str:
        pattern = r"film/(\d+)(?:-[^/]+)?(?:/(\d+))?"
        matches = re.search(pattern, url)
        if matches:
            movie_id = matches.group(1)
            episode_id = matches.group(2)
            return f"{movie_id}_{episode_id}" if episode_id else movie_id
        LOGGER.warning(f"Failed to parse movie ID from URL '{url}'")
        return 'UNKNOWN'

    @classmethod
    def _parse_review_count(cls, text: str) -> int:
        for space in (' ', '\xa0'):
            text = text.replace(space, '')
        return int(text.split('rec', maxsplit=1)[0])


