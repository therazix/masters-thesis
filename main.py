import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import click

import dataset_parser
import models.ensemble
import models.mistral
import models.xlm_roberta
import scrapers.csfd
import scrapers.reddit
import scrapers.tn_cz

TEMP_DIR = Path(tempfile.gettempdir()) / 'DP_541699'


def to_path(ctx, param, value):
    return Path(value) if value else None


### Scraping commands ###

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


@click.command(name='tn-cz')
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
def scrape_tncz(ctx: click.Context, output_dir: Optional[Path], user_agent: Optional[str],
                resume: bool):
    logger = ctx.obj['logger']
    tncz_scraper = scrapers.tn_cz.TNCZScraper(TEMP_DIR / 'tn_cz', output_dir, user_agent, logger)
    if resume:
        tncz_scraper.load_state()
    tncz_scraper.scrape()


@click.group()
def scrape():
    pass


scrape.add_command(scrape_reddit)
scrape.add_command(scrape_csfd)
scrape.add_command(scrape_tncz)


### Training commands ###

@click.command(name='xlm-roberta')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for model outputs during training (e.g. checkpoints).')
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
@click.option('-e', '--epochs',
              required=True,
              type=int,
              help='Number of epochs for training.')
@click.option('--testing-set',
              required=False,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model. If not provided, final evaluation is skipped.')
@click.option('--checkpoint',
              required=False,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Checkpoint to resume training. Must be a directory.')
@click.pass_context
def train_xlm_roberta(ctx: click.Context, output_dir: Path, training_set: Path,
                      validation_set: Path, epochs: int, testing_set: Optional[Path] = None,
                      checkpoint: Optional[Path] = None):
    logger = ctx.obj['logger']
    xlm_roberta = models.xlm_roberta.XLMRoberta.for_training(
        output_dir, training_set, validation_set, testing_set, checkpoint, logger)
    xlm_roberta.train(epochs=epochs)


@click.command(name='ensemble')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for model outputs during training (e.g. checkpoints).')
@click.option('--model',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Path to the trained XLM-RoBERTa model.')
@click.option('--training-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Training set for the model.')
@click.option('--testing-set',
              required=False,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model. If not provided, final evaluation is skipped.')
@click.pass_context
def train_ensemble(ctx: click.Context, output_dir: Path, model: Path, training_set: Path,
                   testing_set: Optional[Path] = None):
    logger = ctx.obj['logger']
    xlm_roberta = models.xlm_roberta.XLMRoberta.for_testing(
        output_dir, model, testing_set, logger)
    ensemble = models.ensemble.Ensemble(xlm_roberta, logger)
    ensemble.train(output_dir, training_set, testing_set)


@click.group()
def train():
    pass


train.add_command(train_xlm_roberta)
train.add_command(train_ensemble)


### Testing commands ###

@click.command(name='xlm-roberta')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for model outputs during training (e.g. checkpoints).')
@click.option('--model',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Path to a saved model. Must be a directory.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.pass_context
def test_xlm_roberta(ctx: click.Context, output_dir: Path, model: Path, testing_set: Path):
    logger = ctx.obj['logger']
    xlm_roberta = models.xlm_roberta.XLMRoberta.for_testing(output_dir, model, testing_set, logger)
    xlm_roberta.test()


@click.command(name='ensemble')
@click.option('--model',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Path to the trained XLM-RoBERTa model.')
@click.option('--classifiers-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Path to the directory with trained classifiers.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.pass_context
def test_ensemble(ctx: click.Context, model: Path, classifiers_dir: Path, testing_set: Path):
    logger = ctx.obj['logger']
    xlm_roberta = models.xlm_roberta.XLMRoberta.for_testing(Path('.'), model, testing_set, logger)
    ensemble = models.ensemble.Ensemble(xlm_roberta, logger)
    ensemble.test(classifiers_dir, testing_set)


@click.command(name='mistral')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.option('--lang',
              required=False,
              type=str,
              default='cz',
              help="What language to use for the model's instructions. Either 'cz' or 'en'.")
@click.option('--crop',
              required=False,
              is_flag=True,
              help='Crop all texts from dataset to fit the model input size. '
                   'If not provided, texts that are too long will be skipped.')
@click.option('--reps',
              required=False,
              type=int,
              default=3,
              show_default=True,
              help='Number of repetitions for testing.')
@click.option('--token',
              required=False,
              type=str,
              help='Hugging Face API token. If not provided, HF_TOKEN environment '
                   'variable will be used.')
@click.pass_context
def test_mistral(ctx: click.Context, testing_set: Path, lang: str, crop: bool, reps: int, token: Optional[str] = None):
    logger = ctx.obj['logger']
    if reps < 1:
        raise ValueError('Number of repetitions must be at least 1')
    mistral = models.mistral.Mistral(testing_set, lang, crop, token, logger)
    mistral.test(reps)


@click.group()
def test():
    pass


test.add_command(test_xlm_roberta)
test.add_command(test_ensemble)
test.add_command(test_mistral)


### Other commands ###

@click.command(name='create')
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
def dataset_create(ctx: click.Context, input_file: Path, output_dir: Optional[Path],
                   num_of_authors: int, add_out_of_class: bool, add_text_features: bool):
    logger = ctx.obj['logger']
    output_dir = output_dir or Path('datasets')
    parser = dataset_parser.DatasetParser(input_file, logger)
    parser.create(output_dir, num_of_authors, add_out_of_class, add_text_features, (0.7, 0.15, 0.15))


@click.command(name='info')
@click.option('-i', '--input-file',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Path to the dataset.')
@click.pass_context
def dataset_info(ctx: click.Context, input_file: Path):
    logger = ctx.obj['logger']
    parser = dataset_parser.DatasetParser(input_file, logger)
    parser.info()


@click.group()
def dataset():
    pass

dataset.add_command(dataset_create)
dataset.add_command(dataset_info)


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

    if verbose >= 1:
        # Print log messages to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formater)
        logger.addHandler(console_handler)
    if verbose >= 2:
        # Log also debug messages
        logger.setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj['logger'] = logger


cli.add_command(scrape)
cli.add_command(train)
cli.add_command(test)
cli.add_command(dataset)

if __name__ == "__main__":
    cli()
