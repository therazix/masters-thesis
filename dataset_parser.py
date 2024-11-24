import json
import logging
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import tiktoken
from pandarallel import pandarallel

import copy

import models.gpt_4o
import prompts
import json
from models.gpt_4o import GPT4o
from utils import clean_text, get_child_logger

pd.options.mode.chained_assignment = None

CHARS = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
PUNCTUATION = '.:,?!-;('
DIACRITICS = 'áčďéěíňóřšťúůýž'

def _init_pandarallel():
    from warnings import simplefilter
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    pandarallel.initialize()

def _extract_style(text: str):
    words = text.split()

    len_text = len(text)  # Number of characters in the whole text
    len_words = len(words)  # Number of words
    avg_len = np.mean([len(w) for w in words])  # Average word length
    num_short_w = len([w for w in words if len(w) < 3])  # Number of words with less than 3 chars
    per_digit = sum(c.isdigit() for c in text) / len_text  # Percentage of digits
    per_cap = sum(1 for c in text if c.isupper()) / len_text  # Percentage of capital letters
    richness = len(list(set(words))) / len_words  # Ratio of unique words to all words
    # Frequencies of each character relative to the whole text
    all_chars = CHARS + DIGITS + DIACRITICS
    frequencies = {char: sum(1 for c in text if c.lower() == char) / len_text for char in all_chars}

    return pd.Series(
        [avg_len, len_text, len_words, num_short_w, per_digit, per_cap, richness, *frequencies.values()]
    )

def _insert_features(df: pd.DataFrame):
    columns = ['avg_len', 'len_text', 'len_words', 'num_short_w', 'per_digit', 'per_cap', 'richness']
    columns += [f'cf_{c}' for c in CHARS + DIGITS + DIACRITICS]
    df[columns] = df['text'].parallel_apply(lambda x: _extract_style(x))


class DatasetParser:
    def __init__(self,
                 input_file: Path,
                 logger: Optional[logging.Logger] = None):
        self.output_dir = Path('datasets/')
        self.input_file = input_file.resolve()
        self.logger = get_child_logger(__name__, logger)
        self.logger.debug(f"Reading input file '{self.input_file}'")
        df = pd.read_csv(self.input_file)

        if 'text' not in df.columns:
            raise ValueError("Input file must have a 'text' column")

        if 'author' in df.columns:
            df = df[['author', 'text']]
            df = df.rename(columns={'author': 'label'})

        if 'label' not in df.columns:
            raise ValueError("Input file must have a 'label' or 'author' column")

        self.df = df.drop_duplicates()

        # Get all author names sorted by the number of texts
        self.authors = self.df['label'].value_counts().index.tolist()

    def info(self, top: List[int]):
        info = {
            'total_num_of_texts': len(self.df),
            'total_num_of_authors': len(self.authors),
            'total_avg_text_length': round(self.df['text'].apply(len).mean(), 2),
            'total_min_text_length': self.df['text'].apply(len).min(),
            'total_max_text_length': self.df['text'].apply(len).max(),
            'total_avg_texts_per_author': round(self.df['label'].value_counts().mean(), 2),
            'total_min_texts_per_author': self.df['label'].value_counts().min(),
            'total_max_texts_per_author': self.df['label'].value_counts().max(),
        }
        for i in top:
            top_authors = self.df['label'].value_counts().index[:i]
            info[f'top{i}_num_texts'] = self.df['label'].value_counts().loc[top_authors].sum()
            info[f'top{i}_avg_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).mean()  # noqa
            info[f'top{i}_min_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).min()  # noqa
            info[f'top{i}_max_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).max()  # noqa
            info[f'top{i}_avg_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().mean()  # noqa
            info[f'top{i}_min_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().min()  # noqa
            info[f'top{i}_max_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().max()  # noqa

        self.logger.info('Dataset information:')
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")

    def create(self,
               output_dir: Path,
               num_of_authors: int,
               limit: Optional[int] = None,
               add_out_of_class: bool = False,
               add_text_features: bool = False,
               split: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        if num_of_authors < 1:
            raise ValueError('Number of authors must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('Limit must be at least 1')
        if sum(split) != 1:
            raise ValueError('Sum of split values must be 1')

        self.logger.debug('Creating dataset...')
        _init_pandarallel()

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        selected_authors = self.authors[:num_of_authors]
        non_selected_authors = self.authors[num_of_authors:]

        # Initialize the train and test dataframes
        train = pd.DataFrame(columns=['label', 'text'])
        val = pd.DataFrame(columns=['label', 'text'])
        test = pd.DataFrame(columns=['label', 'text'])

        self.df['text'] = self.df['text'].parallel_apply(lambda x: clean_text(x))

        for author in selected_authors:
            author_df = self.df[self.df['label'] == author]  # Get all texts of the author
            if limit is not None:
                author_df = author_df.head(limit)
                if len(author_df) < limit:
                    self.logger.warning(f"Author '{author}' has less than {limit} texts")
            author_df = author_df.sample(frac=1).reset_index(drop=True)  # Shuffle author's texts
            # Get the indexes for splitting the author's texts into train, validation and test sets
            split_indexes = [int(sum(split[:i+1]) * len(author_df)) for i in range(len(split) - 1)]

            train_df = author_df.iloc[:split_indexes[0]]
            val_df = author_df.iloc[split_indexes[0]:split_indexes[1]]
            test_df = author_df.iloc[split_indexes[1]:]

            # Add author's texts to the train and test dataframes
            train = pd.concat([train, train_df], ignore_index=True)
            val = pd.concat([val, val_df], ignore_index=True)
            test = pd.concat([test, test_df], ignore_index=True)

        if add_out_of_class:
            self.logger.info('Adding out-of-class texts')
            out_of_class_df = pd.DataFrame(columns=['label', 'text'])
            last_author_text_count = len(self.df[self.df['label'] == selected_authors[-1]])
            # math.ceil is used to ensure that the number of texts per author is at least 1
            texts_per_author = math.ceil(last_author_text_count / len(non_selected_authors))
            for author in non_selected_authors:
                author_df = self.df[self.df['label'] == author]
                author_df = author_df.sample(frac=1).reset_index(drop=True)
                author_df = author_df.iloc[:texts_per_author]
                out_of_class_df = pd.concat([out_of_class_df, author_df], ignore_index=True)

            out_of_class_df['label'] = str(len(selected_authors))
            out_of_class_df = out_of_class_df.sample(frac=1).reset_index(drop=True)
            split_indexes = [int(sum(split[:i + 1]) * len(out_of_class_df)) for i in range(len(split) - 1)]  # noqa

            train_out_of_class = out_of_class_df.iloc[:split_indexes[0]]
            val_out_of_class = out_of_class_df.iloc[split_indexes[0]:split_indexes[1]]
            test_out_of_class = out_of_class_df.iloc[split_indexes[1]:]

            train = pd.concat([train, train_out_of_class], ignore_index=True)
            val = pd.concat([val, val_out_of_class], ignore_index=True)
            test = pd.concat([test, test_out_of_class], ignore_index=True)

        # Rename all authors
        replace_dict = {author: str(i) for i, author in enumerate(self.authors)}
        train['label'] = train['label'].replace(replace_dict)
        val['label'] = val['label'].replace(replace_dict)
        test['label'] = test['label'].replace(replace_dict)

        # Shuffle the dataframes
        train = train.sample(frac=1).reset_index(drop=True)
        val = val.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        if add_text_features:
            self.logger.info('Adding text features')
            _insert_features(train)
            _insert_features(val)
            _insert_features(test)

        ooc_suffix = '_withOOC' if add_out_of_class else ''
        train.to_csv(self.output_dir / f'train_top{num_of_authors}{ooc_suffix}.csv', index=False)
        val.to_csv(self.output_dir / f'val_top{num_of_authors}{ooc_suffix}.csv', index=False)
        test.to_csv(self.output_dir / f'test_top{num_of_authors}{ooc_suffix}.csv', index=False)
        self.logger.info(f"Dataset created and saved to '{self.output_dir}'")

    def create_prompting(self, output_dir: Path, num_of_authors: int, reps: int = 3):
        if reps < 1:
            raise ValueError('Number of repetitions must be at least 1')
        if num_of_authors < 2:
            raise ValueError('Number of authors must be at least 2')

        self.logger.debug('Creating dataset for prompting...')

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _init_pandarallel()

        self.df['text'] = self.df['text'].parallel_apply(lambda x: clean_text(x))

        # Filter out texts that are too short
        self.df = self.df[self.df['text'].parallel_apply(lambda x: _has_minimum_length(x))]

        columns = ['label', 'query_text', 'example_text']
        results = []
        for _ in range(reps):
            authors = random.sample(self.authors, num_of_authors)
            authors_texts = []
            for i, author in enumerate(authors):
                try:
                    text_1, text_2 = self.df[self.df['label'] == author]['text'].sample(2)
                except ValueError as err:
                    raise ValueError(f'Not enough samples for author {author}.') from err
                authors_texts.append([i, text_1, text_2])
            results.append(pd.DataFrame(authors_texts, columns=columns))
        results_df = pd.concat(results, keys={i: res for i, res in enumerate(results)})
        results_df = results_df.reset_index(level=1, drop=True)

        filename = f'test_prompts_{num_of_authors}authors_{reps}reps.csv'
        results_df.to_csv(self.output_dir / filename, index=True)
        self.logger.info(f"Dataset created and saved to '{self.output_dir}'")


    def create_finetuning(self, output_dir: Path, num_of_authors: int, reps: int):
        if num_of_authors < 2:
            raise ValueError('Number of authors must be at least 2')

        self.logger.debug('Creating dataset for fine-tuning...')

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _init_pandarallel()

        self.df['text'] = self.df['text'].parallel_apply(lambda x: clean_text(x))

        # Filter out texts that are too short
        self.df = self.df[self.df['text'].parallel_apply(lambda x: _has_minimum_length(x))]

        results = pd.DataFrame(columns=['label', 'query_text', 'example_text'])
        for _ in range(reps):
            authors = random.sample(self.authors, num_of_authors)
            authors_texts = []
            for i, author in enumerate(authors):
                try:
                    text_1, text_2 = self.df[self.df['label'] == author]['text'].sample(2)
                except ValueError as err:
                    raise ValueError(f'Not enough samples for author {author}.') from err
                authors_texts.append([i, text_1, text_2])

            authors_df = pd.DataFrame(authors_texts, columns=['label', 'query_text', 'example_text'])

            examples = json.dumps(
                {row['label']: row['example_text'] for _, row in authors_df.iterrows()},
                ensure_ascii=False
            )

            authors_df['example_text'] = examples
            results = pd.concat([results, authors_df], ignore_index=True)

        results = results.sample(frac=1).reset_index(drop=True)

        filename = f'finetuning_{num_of_authors}authors_{reps}reps.jsonl'
        with (self.output_dir / filename).open('w', encoding='utf-8') as f:
            for i, row in results.iterrows():
                messages = copy.deepcopy(prompts.prompts_finetuning)
                messages[-1]['content'] = messages[-1]['content'].format(
                    query_text=row['query_text'],
                    example_text=row['example_text'],
                    correct_author=row['label']
                )
                json_row = {
                    'custom_id': f'request_{i}',
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': 'gpt-4o-mini-2024-07-18',
                        'messages': messages,
                        'max_tokens': 1000,
                        'temperature': 0.7

                    }
                }
                f.write(json.dumps(json_row, ensure_ascii=False) + '\n')

        self.logger.info(f"Dataset created and saved to '{self.output_dir}'")


def _has_minimum_length(text: str) -> bool:
    encoding = tiktoken.encoding_for_model('gpt-4o')
    count = len(encoding.encode(text))
    return count > 60


# def complete_finetune_data(dataset_path: Path, openai_api_key: Optional[str] = None):
#     client = GPT4o.get_client(openai_api_key)
#     df = pd.read_csv(dataset_path)
#     results_df = df.copy()
#     for i, row in df.iterrows():
#         query = row['query_text']
#         examples = row['example_text']
#         correct_author = row['label']
#         messages = copy.deepcopy(prompts.prompts_finetuning)
#         messages[-1]['content'] = messages[-1]['content'].format(query=query,
#                                                                  examples=examples,
#                                                                  correct_author=correct_author)
#         try:
#             completion = client.chat.completions.create(
#                 model=models.gpt_4o.MODEL_NAME,
#                 messages=messages,
#                 max_tokens=800,
#                 temperature=0.0,
#             )
#             response = completion.choices[0].message
#             if response.parsed:
#                 result = cast(GPTResponse, response.parsed)
#             elif response.content:
#                 result = response.content
#             else:
#                 result = 'error'
#         except Exception:
#             result = 'error'
#
#         results_df.loc[i, 'finetune_response'] = result
#
#     results_df.to_csv(dataset_path.parent / (dataset_path.stem + '_complete.csv'))

