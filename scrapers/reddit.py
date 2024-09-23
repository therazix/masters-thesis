import os
from dotenv import load_dotenv
import praw
import csv
import utils
import time
import logging
from praw import models
from typing import Optional
from pathlib import Path
from . import Scraper

RATELIMIT_SECONDS = 300
INVALID_AUTHORS = ['automoderator', 'moderator', 'deleted', '[deleted]', 'removed', '[removed]']
LOGGER = logging.getLogger(__name__)


class RedditScraper(Scraper):
    def __init__(self, output_dir: Optional[Path] = None, user_agent: Optional[str] = None):
        output_dir = output_dir or Path('scraped_data/reddit')
        output_dir = output_dir.resolve()
        super().__init__(output_dir, user_agent)
        self._init_reddit()

    def _init_reddit(self):
        load_dotenv()
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        username = os.getenv('REDDIT_USERNAME')
        password = os.getenv('REDDIT_PASSWORD')

        if not client_id or not client_secret or not username or not password:
            raise ValueError("Reddit API credentials are missing. Please set REDDIT_CLIENT_ID, "
                             "REDDIT_CLIENT_SECRET, REDDIT_USERNAME, and REDDIT_PASSWORD "
                             "environment variables.")

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            password=password,
            user_agent=self.user_agent,
            username=username,
            ratelimit_seconds=RATELIMIT_SECONDS
        )

    @classmethod
    def _is_valid_author(cls, author: models.Redditor) -> bool:
        if not hasattr(author, 'name'):
            return False
        if not author.name or author.name.lower() in INVALID_AUTHORS:
            return False
        return not author.name.endswith('bot')

    @classmethod
    def _is_valid_comment(cls, comment: models.Comment) -> bool:
        if not hasattr(comment, 'author') or not hasattr(comment, 'body'):
            return False
        if not cls._is_valid_author(comment.author):
            return False
        if not comment.body:
            return False
        return True

    @classmethod
    def _parse_reddit_post(cls, text: str) -> Optional[str]:
        text = utils.remove_markdown(text)
        return cls._parse_author_text(text)

    @classmethod
    def _parse_replies(cls, writer: csv.writer, comment: models.Comment):
        for reply in comment.replies:
            if cls._is_valid_comment(reply):
                text = cls._parse_reddit_post(reply.body)
                if text:
                    writer.writerow((reply.author.name, reply.id, text))
                cls._parse_replies(writer, reply)

    @classmethod
    def _submission_generator(cls, subreddit: models.Subreddit):
        def fetch_submissions(generator: models.ListingGenerator):
            for submission in generator:
                if submission.id not in processed_submissions:
                    processed_submissions.add(submission.id)
                    yield submission

        processed_submissions = set()
        generators = [
            subreddit.top(limit=None, time_filter='all'),
            subreddit.top(limit=None, time_filter='year'),
            subreddit.top(limit=None, time_filter='month'),
            subreddit.hot(limit=None)
        ]

        for generator in generators:
            assert isinstance(generator, models.ListingGenerator)
            yield from fetch_submissions(generator)

    def scrape(self):
        self._init_directories()

        subreddit = self.reddit.subreddit("czech")

        filepath = self.output_dir / f'reddit_{time.strftime("%y%m%d_%H%M%S")}.csv'
        with filepath.open('a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('author', 'post_id', 'text'))  # Header
            for submission in self._submission_generator(subreddit):
                if self._is_valid_author(submission.author) and submission.selftext:
                    text = self._parse_reddit_post(submission.selftext)
                    if text:
                        writer.writerow((submission.author.name, submission.id, text))
                for comment in submission.comments:
                    if self._is_valid_comment(comment):
                        text = self._parse_reddit_post(comment.body)
                        if text:
                            writer.writerow((comment.author.name, comment.id, text))
                        self._parse_replies(writer, comment)
                LOGGER.info(f"Finished scraping of submission '{submission.id}'")
