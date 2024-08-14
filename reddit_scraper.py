import os
import praw
from praw import models
import time
import csv
import utils
from pathlib import Path


DATA_PATH = Path('data/reddit').resolve()

INVALID_AUTHORS = ['automoderator', 'moderator', 'deleted', '[deleted]', 'removed', '[removed]']


def init_directories():
    DATA_PATH.mkdir(parents=True, exist_ok=True)


def initialize_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        password=os.getenv('REDDIT_CLIENT_PASSWORD'),
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        username=os.getenv('REDDIT_CLIENT_USERNAME'),
        ratelimit_seconds=300
    )


def is_valid_author(author: models.Redditor) -> bool:
    if not hasattr(author, 'name'):
        return False
    return author.name and author.name.lower() not in INVALID_AUTHORS


def is_valid_comment(comment: models.Comment) -> bool:
    if not hasattr(comment, 'author') or not hasattr(comment, 'body'):
        return False
    if not hasattr(comment.author, 'name'):
        return False
    if not comment.author.name or comment.author.name.lower() in INVALID_AUTHORS:
        return False
    if not comment.body:
        return False
    return True


def parse_reddit_post(text: str) -> str | None:
    text = utils.remove_markdown(text)
    return utils.parse_author_text(text)


def parse_replies(writer: csv.writer, comment: models.Comment):
    for reply in comment.replies:
        if is_valid_comment(reply):
            text = parse_reddit_post(reply.body)
            if text:
                writer.writerow([reply.author.name, reply.id, text])
            parse_replies(writer, reply)


def submission_generator(subreddit: models.Subreddit):
    for submission in subreddit.top(limit=None, time_filter='all'):
        yield submission
    for submission in subreddit.top(limit=None, time_filter='year'):
        yield submission
    for submission in subreddit.top(limit=None, time_filter='month'):
        yield submission
    for submission in subreddit.hot(limit=None):
        yield submission


def main():
    init_directories()

    reddit = initialize_reddit()
    subreddit = reddit.subreddit("czech")

    filename = DATA_PATH / f'reddit_{time.strftime("%y%m%d%H%M%S")}.csv'

    processed_submissions = set()

    with filename.open('a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['author', 'post_id', 'text'])  # Header
        for submission in submission_generator(subreddit):
            if submission.id in processed_submissions:
                continue
            if is_valid_author(submission.author) and submission.selftext:
                text = parse_reddit_post(submission.selftext)
                if text:
                    writer.writerow([submission.author.name, submission.id, text])
            for comment in submission.comments:
                if is_valid_comment(comment):
                    text = parse_reddit_post(comment.body)
                    if text:
                        writer.writerow([comment.author.name, comment.id, text])
                    parse_replies(writer, comment)
            processed_submissions.add(submission.id)
            print(f'Parsed submission: {submission.title}')


if __name__ == "__main__":
    main()
