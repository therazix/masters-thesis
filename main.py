import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import click

import prepare_dataset
import scrapers.csfd
import scrapers.reddit
import models.xlm_roberta
import utils

TEMP_DIR = Path(tempfile.gettempdir()) / 'DP_541699'


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
    reddit_scraper = scrapers.reddit.RedditScraper(output_dir, user_agent)
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
@click.option('--resume',
              required=False,
              is_flag=True,
              help='Continue scraping from the last saved state.')
def scrape_csfd(output_dir: Optional[Path], user_agent: Optional[str], resume: bool):
    csfd_scraper = scrapers.csfd.CSFDScraper(TEMP_DIR / 'csfd', output_dir, user_agent)
    if resume:
        csfd_scraper.load_state()
    csfd_scraper.scrape()


@click.group()
def scrape():
    pass


scrape.add_command(scrape_reddit)
scrape.add_command(scrape_csfd)


@click.command(name='xlm-roberta')
@click.option('--training-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Training set for the model.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.option('--checkpoint-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for model checkpoints.')
@click.option('--model-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for saved model.')
@click.option('-e', '--epochs',
              required=True,
              type=int,
              help='Number of epochs for training.')
@click.option('--resume',
              required=False,
              is_flag=True,
              help='Continue training from the last checkpoint.')
def train_xlm_roberta(training_set: Path, testing_set: Path, checkpoint_dir: Path,
                      model_dir: Path, epochs: int, resume: bool):
    xlm_roberta = models.xlm_roberta.XLMRoberta(
        training_set, testing_set, checkpoint_dir, model_dir)
    xlm_roberta.train(epochs=epochs, resume_training=resume)


@click.group()
def train():
    pass


train.add_command(train_xlm_roberta)


@click.command(name='create-dataset')
@click.option('-i', '--input-file',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Input file with scraped data.')
@click.option('-o', '--output-dir',
              required=False,
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              callback=to_path,
              help='Output directory for processed dataset.')
@click.option('-n', '--num-of-authors',
              required=True,
              type=int,
              help='Number of authors to extract. Authors with the most texts are selected.')
@click.option('-s', '--train-test-split',
              required=False,
              type=float,
              default=0.75,
              help='Ratio of texts used for training. Default is 0.75.')
@click.option('--add-out-of-class',
              required=False,
              is_flag=True,
              help='Add additional class with out-of-class texts.')
@click.option('--add-text-features',
              required=False,
              is_flag=True,
              help='Add text features to the dataset.')
def create_dataset(input_file: Path, output_dir: Optional[Path], num_of_authors: int,
                   train_test_split: float, add_out_of_class: bool, add_text_features: bool):
    output_dir = output_dir or Path('datasets')
    prepare_dataset.create(input_file, output_dir, num_of_authors, add_out_of_class,
                           add_text_features, train_test_split)


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose: int):
    """TODO: Add a docstring."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (TEMP_DIR / f'{time.strftime("%y%m%d_%H%M%S")}.log').resolve()

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
cli.add_command(train)
cli.add_command(create_dataset)

if __name__ == "__main__":
    cli()
