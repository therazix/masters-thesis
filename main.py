import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List

import click

import dataset_parser
import models.ensemble
import models.gpt_4o
import models.llama3
import models.mistral
import models.xlm_roberta
import scrapers.csfd
import scrapers.reddit
import scrapers.tn_cz

TEMP_DIR = Path(tempfile.gettempdir()) / 'DP_541699'


def to_path(ctx, param, value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(Path(v) for v in value)
    if isinstance(value, list):
        return [Path(v) for v in value]
    return Path(value)


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
@click.option('--limit',
              required=False,
              type=int,
              help='Maximum number of articles to scrape per user.')
@click.pass_context
def scrape_tncz(ctx: click.Context, output_dir: Optional[Path], user_agent: Optional[str],
                resume: bool, limit: Optional[int]):
    logger = ctx.obj['logger']
    tncz_scraper = scrapers.tn_cz.TNCZScraper(TEMP_DIR / 'tn_cz', output_dir, user_agent, logger)
    if resume:
        tncz_scraper.load_state()
    if limit is not None and limit < 1:
        limit = None
    tncz_scraper.scrape(limit)


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
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for outputs during testing.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.option('--model-name',
              required=False,
              type=str,
              default='unsloth/mistral-7b-instruct-v0.3',
              help='Name of the model to test (Hugging Face).')
@click.option('--template',
              required=True,
              type=str,
              help="What prompt template to use for the model's instructions. "
                   "Either 'en', 'cz', 'cz-1shot' or 'cz-inference'.")
@click.option('--token',
              required=False,
              type=str,
              help='Hugging Face API token. If not provided, HF_TOKEN environment '
                   'variable will be used.')
@click.pass_context
def test_mistral(ctx: click.Context, output_dir: Path, testing_set: Path, model_name: str,
                   template: str, token: Optional[str] = None):
    logger = ctx.obj['logger']
    mistral = models.mistral.Mistral(output_dir, testing_set, model_name, template, token, logger)
    mistral.test()


@click.command(name='llama3')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for outputs during testing.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.option('--model-name',
              required=False,
              type=str,
              default='unsloth/Meta-Llama-3.1-8B-Instruct',
              help='Name of the model to test (Hugging Face).')
@click.option('--template',
              required=True,
              type=str,
              help="What prompt template to use for the model's instructions. "
                   "Either 'en', 'cz', 'cz-1shot' or 'cz-inference'.")
@click.option('--token',
              required=False,
              type=str,
              help='Hugging Face API token. If not provided, HF_TOKEN environment '
                   'variable will be used.')
@click.pass_context
def test_llama3(ctx: click.Context, output_dir: Path, testing_set: Path, model_name: str,
                   template: str, token: Optional[str] = None):
    logger = ctx.obj['logger']
    llama3 = models.llama3.Llama3(output_dir, testing_set, model_name, template, token, logger)
    llama3.test()


@click.command(name='gpt4o')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for outputs during testing.')
@click.option('--testing-set',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Testing set for the model.')
@click.option('--template',
              required=True,
              type=str,
              help="What prompt template to use for the model's instructions. "
                   "Either 'en', 'cz' or 'cz-1shot'.")
@click.option('--openai-api-key',
              required=False,
              type=str,
              help='OpenAI API key. If not provided, OPENAI_API_KEY environment '
                   'variable will be used.')
@click.pass_context
def test_gpt4o(ctx: click.Context, output_dir: Path, testing_set: Path, template: str,
               openai_api_key: Optional[str] = None):
    logger = ctx.obj['logger']
    gpt4o = models.gpt_4o.GPT4o(output_dir, testing_set, template, openai_api_key, logger)
    gpt4o.test()


@click.group()
def test():
    pass


test.add_command(test_xlm_roberta)
test.add_command(test_ensemble)
test.add_command(test_mistral)
test.add_command(test_llama3)
test.add_command(test_gpt4o)


### Finetuning commands ###

@click.command(name='mistral')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for outputs during testing.')
@click.option('--model-name',
              required=False,
              type=str,
              default='unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
              help='Name of the model to finetune (Hugging Face).')
@click.option('--repo-id',
              required=True,
              type=str,
              help='Hugging Face model repository ID for saving the finetuned model.')
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
@click.option('--template',
              required=True,
              type=str,
              help="What prompt template to use for the model's instructions. "
                   "Either 'en', 'cz', 'cz-1shot' or 'cz-inference'.")
@click.option('--epochs',
              required=False,
              type=int,
              default=6,
              help='Number of epochs for finetuning.')
@click.option('--token',
              required=False,
              type=str,
              help='Hugging Face API token. If not provided, HF_TOKEN environment '
                   'variable will be used.')
@click.pass_context
def finetune_mistral(ctx: click.Context, output_dir: Path, model_name: str, repo_id: str,
                    training_set: Path, testing_set: Path, template: str, epochs: int,
                    token: Optional[str] = None):
    logger = ctx.obj['logger']
    mistral = models.mistral.Mistral(output_dir, testing_set, model_name, template, token, logger)
    mistral.finetune(str(training_set), repo_id, epochs)
    mistral.test()


@click.command(name='llama3')
@click.option('-o', '--output-dir',
              required=True,
              type=click.Path(file_okay=False, dir_okay=True, readable=True),
              callback=to_path,
              help='Directory for outputs during testing.')
@click.option('--model-name',
              required=False,
              type=str,
              default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
              help='Name of the model to finetune (Hugging Face).')
@click.option('--repo-id',
              required=True,
              type=str,
              help='Hugging Face model repository ID for saving the finetuned model.')
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
@click.option('--template',
              required=True,
              type=str,
              help="What prompt template to use for the model's instructions. "
                   "Either 'en', 'cz', 'cz-1shot' or 'cz-inference'.")
@click.option('--epochs',
              required=False,
              type=int,
              default=6,
              help='Number of epochs for finetuning.')
@click.option('--token',
              required=False,
              type=str,
              help='Hugging Face API token. If not provided, HF_TOKEN environment '
                   'variable will be used.')
@click.pass_context
def finetune_llama3(ctx: click.Context, output_dir: Path, model_name: str, repo_id: str,
                    training_set: Path, testing_set: Path, template: str, epochs: int,
                    token: Optional[str] = None):
    logger = ctx.obj['logger']
    llama3 = models.llama3.Llama3(output_dir, testing_set, model_name, template, token, logger)
    llama3.finetune(str(training_set), repo_id, epochs)
    llama3.test()

@click.group()
def finetune():
    pass


finetune.add_command(finetune_mistral)
finetune.add_command(finetune_llama3)


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
@click.option('-l', '--limit',
              required=False,
              type=int,
              default=None,
              help='Maximum number of texts per author. If not provided, all texts are used.')
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
                   num_of_authors: int, limit: Optional[int], add_out_of_class: bool,
                   add_text_features: bool):
    logger = ctx.obj['logger']
    output_dir = output_dir or Path('datasets')
    parser = dataset_parser.DatasetParser(input_file, logger)
    parser.create(output_dir, num_of_authors, limit, add_out_of_class, add_text_features, (0.7, 0.15, 0.15))


@click.command(name='info')
@click.option('-i', '--input-file',
              required=True,
              type=click.Path(file_okay=True, dir_okay=False, readable=True),
              callback=to_path,
              help='Path to the dataset.')
@click.option('-t', '--top',
              multiple=True,
              type=int,
              default=[],
              help='Show also information for the top N authors.')
@click.pass_context
def dataset_info(ctx: click.Context, input_file: Path, top: List[int]):
    logger = ctx.obj['logger']
    parser = dataset_parser.DatasetParser(input_file, logger)
    top = list(set([t for t in top if t > 0]))
    top.sort()
    parser.info(top)


@click.command(name='create-finetuning')
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
@click.option('-r', '--reps',
                default=3,
                type=int,
                help='Number of repetitions for each author.')
@click.pass_context
def dataset_create_finetuning(ctx: click.Context, input_file: Path, output_dir: Optional[Path],
                              num_of_authors: int, reps: int):
    logger = ctx.obj['logger']
    output_dir = output_dir or Path('datasets')
    parser = dataset_parser.DatasetParser(input_file, logger)
    parser.create_finetuning(output_dir, num_of_authors, reps)


@click.command(name='create-prompting')
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
@click.option('-r', '--reps',
                default=3,
                type=int,
                help='Number of repetitions for each author.')
@click.pass_context
def dataset_create_prompting(ctx: click.Context, input_file: Path, output_dir: Optional[Path],
                   num_of_authors: int, reps: int):
    logger = ctx.obj['logger']
    output_dir = output_dir or Path('datasets')
    parser = dataset_parser.DatasetParser(input_file, logger)
    parser.create_prompting(output_dir, num_of_authors, reps)

@click.group()
def dataset():
    pass

dataset.add_command(dataset_create)
dataset.add_command(dataset_info)
dataset.add_command(dataset_create_finetuning)
dataset.add_command(dataset_create_prompting)


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
cli.add_command(finetune)
cli.add_command(dataset)

if __name__ == "__main__":
    cli()
