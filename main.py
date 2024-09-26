import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import click

import dataset_parser
import scrapers.csfd
import scrapers.reddit
import models.xlm_roberta

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
@click.pass_context
def scrape_reddit(ctx: click.Context, output_dir: Optional[Path], user_agent: Optional[str]):
    logger = ctx.obj['logger']
    reddit_scraper = scrapers.reddit.RedditScraper(output_dir, user_agent, logger)
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
@click.pass_context
def scrape_csfd(ctx: click.Context, output_dir: Optional[Path],
                user_agent: Optional[str], resume: bool):
    logger = ctx.obj['logger']
    csfd_scraper = scrapers.csfd.CSFDScraper(TEMP_DIR / 'csfd', output_dir, user_agent, logger)
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
@click.option('--validation-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Validation set for the model.')
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
@click.option('--checkpoint',
              required=False,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Checkpoint to resume training. Must be a directory.')
@click.pass_context
def train_xlm_roberta(ctx: click.Context, training_set: Path, validation_set: Path,
                      testing_set: Path, checkpoint_dir: Path, model_dir: Path, epochs: int,
                      checkpoint: Optional[Path]):
    logger = ctx.obj['logger']
    xlm_roberta = models.xlm_roberta.XLMRoberta(
        training_set, validation_set, testing_set, checkpoint_dir, model_dir, checkpoint, logger)
    xlm_roberta.train(epochs=epochs)
    xlm_roberta.evaluate()


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
@click.option('--add-out-of-class',
              required=False,
              is_flag=True,
              help='Add additional class with out-of-class texts.')
@click.option('--add-text-features',
              required=False,
              is_flag=True,
              help='Add text features to the dataset.')
@click.pass_context
def create_dataset(ctx: click.Context, input_file: Path, output_dir: Optional[Path],
                   num_of_authors: int, add_out_of_class: bool, add_text_features: bool):
    logger = ctx.obj['logger']
    output_dir = output_dir or Path('datasets')
    parser = dataset_parser.DatasetParser(input_file, output_dir, logger)
    parser.create(num_of_authors, add_out_of_class, add_text_features, (0.7, 0.15, 0.15))


@click.group()
@click.option('-v', '--verbose', count=True)
@click.pass_context
def cli(ctx: click.Context, verbose: int):
    """TODO: Add a docstring."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (TEMP_DIR / f'{time.strftime("%y%m%d_%H%M%S")}.log').resolve()

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s]: %(message)s')

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formater)
    logger.addHandler(file_handler)  # Log to file

    if verbose > 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formater)
        logger.addHandler(console_handler)  # Log also to stdout

    ctx.ensure_object(dict)
    ctx.obj['logger'] = logger


cli.add_command(scrape)
cli.add_command(train)
cli.add_command(create_dataset)

if __name__ == "__main__":
    cli()
