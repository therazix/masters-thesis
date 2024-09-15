import logging
import scraper.csfd
import scraper.reddit
import tempfile
import click
import sys
import time
from pathlib import Path
from typing import Optional


def to_path(ctx, param, value):
    return Path(value) if value else None


@click.command(name='reddit')
@click.option('-o', '--output-dir',
              required=False,
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              callback=to_path,
              help='Output directory for scraped data.')
@click.option('--user-agent',
              required=False,
              type=str,
              help='Custom user agent.')
def scrape_reddit(output_dir: Optional[Path], user_agent: Optional[str]):
    reddit_scraper = scraper.reddit.RedditScraper(output_dir, user_agent)
    reddit_scraper.scrape()


@click.command(name='csfd')
@click.option('-o', '--output-dir',
              required=False,
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              callback=to_path,
              help='Output directory for scraped data.')
@click.option('--user-agent',
              required=False,
              type=str,
              help='Custom user agent.')
def scrape_csfd(output_dir: Optional[Path], user_agent: Optional[str]):
    csfd_scraper = scraper.csfd.CSFDScraper(output_dir, user_agent)
    csfd_scraper.scrape()


@click.group()
def scrape():
    pass


scrape.add_command(scrape_reddit)
scrape.add_command(scrape_csfd)


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose: int):
    """TODO: Add a docstring."""
    temp_dir = Path(tempfile.gettempdir()) / 'DP_541699'
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_file = (temp_dir / f'{time.strftime("%y%m%d_%H%M%S")}.log').resolve()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s]: %(message)s')

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formater)
    root_logger.addHandler(file_handler)  # Log to file

    if verbose > 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formater)
        root_logger.addHandler(console_handler)  # Log also to stdout


cli.add_command(scrape)


if __name__ == "__main__":
    cli()
