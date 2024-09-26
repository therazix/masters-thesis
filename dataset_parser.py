import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import get_child_logger

pandarallel.initialize()
pd.options.mode.chained_assignment = None

CHARS = 'aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž' + '0123456789' + '.!?"„“,:-();/&'


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
    frequencies = {char: sum(1 for c in text if c.lower() == char) / len_text for char in CHARS}

    return pd.Series(
        [avg_len, len_text, len_words, num_short_w,
         per_digit, per_cap, *frequencies.values(), richness]
    )


def _insert_features(df: pd.DataFrame):
    df[['avg_len', 'len_text', 'len_words', 'num_short_w',
        'per_digit', 'per_cap', 'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6',
        'f_7', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15',
        'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24',
        'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33',
        'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42',
        'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51',
        'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_60',
        'f_61', 'f_62', 'f_63', 'f_64', 'richness']] = df['text'].parallel_apply(
        lambda x: _extract_style(x))  # noqa


class DatasetParser:
    def __init__(self,
                 input_file: Path,
                 output_dir: Path,
                 logger: Optional[logging.Logger] = None):
        self.input_file = input_file.resolve()
        self.output_dir = output_dir.resolve()
        self.logger = get_child_logger(__name__, logger)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.input_file)
        if 'author' not in df.columns or 'text' not in df.columns:
            raise ValueError("Input file must have columns 'author' and 'text'")
        df = df[['author', 'text']]
        df = df.rename(columns={'author': 'label'})
        self.df = df.drop_duplicates()
        # Get all author names sorted by the number of texts
        self.authors = self.df['label'].value_counts().index.tolist()
        self._log_data_info()

    def _log_data_info(self):
        info = {
            'total_num_texts': len(self.df),
            'total_num_authors': len(self.authors),
            'total_avg_text_length': self.df['text'].apply(len).mean(),
            'total_min_text_length': self.df['text'].apply(len).min(),
            'total_max_text_length': self.df['text'].apply(len).max(),
            'total_avg_texts_per_author': self.df['label'].value_counts().mean(),
            'total_min_texts_per_author': self.df['label'].value_counts().min(),
            'total_max_texts_per_author': self.df['label'].value_counts().max(),
        }
        for i in [5, 10, 20, 50]:
            if i > len(self.authors):
                break
            top_authors = self.df['label'].value_counts().index[:i]
            info[f'top{i}_num_texts'] = self.df['label'].value_counts().loc[top_authors].sum()
            info[f'top{i}_avg_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).mean()  # noqa
            info[f'top{i}_min_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).min()  # noqa
            info[f'top{i}_max_text_length'] = self.df[self.df['label'].isin(top_authors)]['text'].apply(len).max()  # noqa
            info[f'top{i}_avg_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().mean()  # noqa
            info[f'top{i}_min_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().min()  # noqa
            info[f'top{i}_max_texts_per_author'] = self.df[self.df['label'].isin(top_authors)]['label'].value_counts().max()  # noqa

        self.logger.info('Data information:')
        for key, value in info.items():
            self.logger.info(f"{key}: {value}")

    def create(self,
               num_of_authors: int,
               add_out_of_class: bool = False,
               add_text_features: bool = False,
               split: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        if num_of_authors < 1:
            raise ValueError('Number of authors must be at least 1')
        if sum(split) != 1:
            raise ValueError('Sum of split values must be 1')

        selected_authors = self.authors[:num_of_authors]
        non_selected_authors = self.authors[num_of_authors:]

        # Initialize the train and test dataframes
        train = pd.DataFrame(columns=['label', 'text'])
        val = pd.DataFrame(columns=['label', 'text'])
        test = pd.DataFrame(columns=['label', 'text'])

        for author in selected_authors:
            author_df = self.df[self.df['label'] == author]  # Get all texts of the author
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
            val_out_od_class = out_of_class_df.iloc[split_indexes[0]:split_indexes[1]]
            test_out_of_class = out_of_class_df.iloc[split_indexes[1]:]

            train = pd.concat([train, train_out_of_class], ignore_index=True)
            val = pd.concat([val, val_out_od_class], ignore_index=True)
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

        ooc_suffix = 'withOOC' if add_out_of_class else 'withoutOOC'
        train.to_csv(self.output_dir / f'train_top{num_of_authors}_{ooc_suffix}.csv', index=False)
        val.to_csv(self.output_dir / f'val_top{num_of_authors}_{ooc_suffix}.csv', index=False)
        test.to_csv(self.output_dir / f'test_top{num_of_authors}_{ooc_suffix}.csv', index=False)
        self.logger.info(f"Dataset created and saved to '{self.output_dir}'")
