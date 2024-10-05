import heapq
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.util import ngrams
from sklearn.linear_model import LogisticRegression

from metrics import compute_metrics
from utils import get_child_logger
from . import xlm_roberta


def process_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dataset_path)
    return df[['label', 'text']], df.drop(columns=['label', 'text'])


def get_common_ngrams(text: str, n: int, count: int = 100):
    n_grams = ngrams(text, n)
    counter = dict(Counter(n_grams))
    return heapq.nlargest(count, counter.keys(), key=lambda k: counter[k])


class Ensemble:
    def __init__(self, model: xlm_roberta.XLMRoberta, logger: Optional[logging.Logger] = None):
        self.logger = get_child_logger(__name__, logger)
        self.model = model

    def _init_train(self, output_dir: Path, training_set: Path, testing_set: Optional[Path] = None):
        self.output_dir = output_dir.resolve()
        self.train_df, self.train_features_df = process_dataset(training_set)
        self.test_df, self.test_features_df = process_dataset(testing_set) if testing_set else (None, None)

        text = ' '.join(self.train_df['text'].values)
        self.bigram_list = get_common_ngrams(text, 2)
        self.trigram_list = get_common_ngrams(text, 3)
        self._init_directories()
        self._save_ngrams()

    def _init_test(self, classifiers_dir: Path, testing_set: Path):
        self.classifiers_dir = classifiers_dir.resolve()
        self.test_df, self.test_features_df = process_dataset(testing_set)
        self._load_ngrams()

    def _save_ngrams(self):
        lists = (self.bigram_list, self.trigram_list)
        filename = self.output_dir / 'common_ngrams.pkl'
        with filename.open('wb') as f:
            pickle.dump(lists, f)

    def _load_ngrams(self):
        filename = self.classifiers_dir / 'common_ngrams.pkl'
        if not filename.exists():
            raise FileNotFoundError(
                "File 'common_ngrams.pkl' not found. Ensure you provided "
                "correct directory with 'classifiers-dir' argument.")
        with open('common_ngrams.pkl', 'rb') as f:
            lists = pickle.load(f)
        self.bigram_list, self.trigram_list = lists

    def _init_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_ngram_frequencies(self, text: str) -> pd.Series:
        result = []
        num_bigrams = len(Counter(zip(text, text[1:])))
        num_trigrams = len(Counter(zip(text, text[1:], text[2:])))

        for ngram in self.bigram_list:
            result.append(text.count(''.join(ngram)) / num_bigrams)

        for ngram in self.trigram_list:
            result.append(text.count(''.join(ngram)) / num_trigrams)

        return pd.Series(result)

    def _get_ngram_frequencies(self, df: pd.DataFrame):
        train_ngrams = df['text'].apply(lambda x: self._calculate_ngram_frequencies(x)).values
        return pd.DataFrame(train_ngrams)

    def _get_model_predictions(self, df: pd.DataFrame) -> np.ndarray:
        output = self.model.predict(Dataset.from_pandas(df), tokenize=True)  # noqa
        return output.predictions

    def _train_style(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.logger.info('Training style classifier')
        lr = LogisticRegression(random_state=0).fit(self.train_features_df, self.train_df['label'])

        # Save model
        filepath = self.output_dir / 'lr_style.pkl'
        with filepath.open('wb') as f:
            pickle.dump(lr, f)

        train_proba = lr.predict_proba(self.train_features_df)

        if self.test_df is not None:
            test_pred = lr.predict(self.test_features_df)
            metrics = compute_metrics(self.test_df['label'], test_pred)
            self.logger.info(f'Style classifier metrics: {metrics}')
            test_proba = lr.predict_proba(self.test_features_df)
            return train_proba, test_proba

        return train_proba, None

    def _train_ngrams(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.logger.info('Training n-grams classifier')
        train_ngrams_df = self._get_ngram_frequencies(self.train_df)

        lr = LogisticRegression(random_state=0).fit(train_ngrams_df, self.train_df['label'])

        # Save model
        filepath = self.output_dir / 'lr_ngrams.pkl'
        with filepath.open('wb') as f:
            pickle.dump(lr, f)

        train_proba = lr.predict_proba(train_ngrams_df)

        if self.test_df is not None:
            test_ngrams_df = self._get_ngram_frequencies(self.test_df)
            test_pred = lr.predict(test_ngrams_df)
            metrics = compute_metrics(self.test_df['label'], test_pred)
            self.logger.info(f'N-grams classifier metrics: {metrics}')
            test_proba = lr.predict_proba(test_ngrams_df)
            return train_proba, test_proba

        return train_proba, None

    def train(self, output_dir: Path, training_set: Path, testing_set: Optional[Path] = None):
        self._init_train(output_dir, training_set, testing_set)

        model_train_proba = self._get_model_predictions(self.train_df)
        model_test_proba = self._get_model_predictions(self.test_df)
        style_train_proba, style_test_proba = self._train_style()
        ngrams_train_proba, ngrams_test_proba = self._train_ngrams()

        self.logger.info('Training final classifier')
        train_combined = np.concatenate(
            [model_train_proba, style_train_proba, ngrams_train_proba], axis=1)

        lr = LogisticRegression(random_state=0).fit(train_combined, self.train_df['label'])

        # Save model
        filepath = self.output_dir / 'lr_final.pkl'
        with filepath.open('wb') as f:
            pickle.dump(lr, f)

        if self.test_df is not None:
            test_combined = np.concatenate(
                [model_test_proba, style_test_proba, ngrams_test_proba], axis=1)
            test_pred = lr.predict(test_combined)
            metrics = compute_metrics(self.test_df['label'], test_pred)
            self.logger.info(f'Final classifier metrics: {metrics}')

    def test(self, classifiers_dir: Path, testing_set: Path):
        self._init_test(classifiers_dir, testing_set)

        with (self.classifiers_dir / 'lr_style.pkl').open('rb') as f:
            lr_style = pickle.load(f)

        with (self.classifiers_dir / 'lr_ngrams.pkl').open('rb') as f:
            lr_ngrams = pickle.load(f)

        with (self.classifiers_dir / 'lr_final.pkl').open('rb') as f:
            lr_final = pickle.load(f)

        model_proba = self._get_model_predictions(self.test_df)
        style_proba = lr_style.predict_proba(self.test_features_df)
        ngrams_proba = lr_ngrams.predict_proba(self._get_ngram_frequencies(self.test_df))

        test_combined = np.concatenate([model_proba, style_proba, ngrams_proba], axis=1)
        test_pred = lr_final.predict(test_combined)
        metrics = compute_metrics(self.test_df['label'], test_pred)
        self.logger.info(f'Final metrics: {metrics}')


