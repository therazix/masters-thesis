import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

from metrics import compute_metrics
from utils import get_child_logger


class LLM:
    def __init__(self,
                 model_name: str,
                 dataset_path: Path,
                 crop: bool,
                 template: Dict[str, str],
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or get_child_logger(__name__)
        self._login(hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.float16,
                                                          attn_implementation='flash_attention_2',
                                                          device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template = template
        self.max_tokens = self.model.config.max_position_embeddings
        self.max_new_tokens = 500
        self.min_tokens_per_text = 56
        df = self._read_dataset(dataset_path)
        num_of_authors = len(df['label'].unique())
        self.max_tokens_per_text = self._get_max_tokens_per_text(self.template, num_of_authors)
        self.dataset = self._parse_dataset(df, crop)


    def _parse_dataset(self, df: pd.DataFrame, crop: bool = False) -> pd.DataFrame:
        if crop:
            result = df[df['text'].apply(lambda x: self._crop_if_needed(x))]
        else:
            result = df[df['text'].apply(lambda x: self._is_token_count_valid(x))]

        min_texts_per_author = 5
        if result.groupby('label').size().min() < min_texts_per_author:
            raise ValueError('Not enough samples per author after dataset parsing. Dataset must '
                             f'have at least {min_texts_per_author} samples per author and each '
                             'text must be within model token limits. If texts are too long, '
                             'consider cropping them using `--crop` flag.')
        return result

    def _count_tokens(self, text: str) -> int:
        encoding = self.tokenizer(text)
        return len(encoding.input_ids)

    def _is_token_count_valid(self, text: str) -> bool:
        count = self._count_tokens(text)
        return self.min_tokens_per_text < count < self.max_tokens_per_text

    def _crop_if_needed(self, text: str) -> str:
        if self._is_token_count_valid(text):
            return text

        sentences = re.split(r'(?<=[.!?]) +', text)
        cropped_text = ''
        for sentence in sentences:
            if self._count_tokens(cropped_text + sentence) > self.max_tokens_per_text:
                break
            cropped_text += sentence + ' '
        return cropped_text.strip()

    def _get_max_tokens_per_text(self, template: Dict[str, str], num_of_examples: int) -> int:
        buffer = 50
        template_len = self._count_tokens(str(template))
        rest = self.max_tokens - self.max_new_tokens - template_len - buffer
        if num_of_examples > 0:
            return rest // num_of_examples
        return rest

    @staticmethod
    def _login(hf_token: Optional[str] = None):
        load_dotenv()
        token = hf_token or os.getenv('HF_TOKEN')
        if not token:
            raise ValueError('Hugging Face API token is missing. Please set HF_TOKEN '
                             'environment variable or pass it as an argument.')
        login(token=token)

    @staticmethod
    def extract_samples(df: pd.DataFrame) -> pd.DataFrame:
        author_names = df['label'].unique().tolist()
        result = pd.DataFrame(columns=['label', 'query_text', 'example_text'])
        for author in author_names:
            # Get 2 random texts from author
            text_1, text_2 = df[df['label'] == author]['text'].sample(2)
            author_df = pd.DataFrame([[author, text_1, text_2]], columns=result.columns)
            result = pd.concat([result, author_df], ignore_index=True)
        result = result.sort_values(by=['label'])
        result = result.reset_index(drop=True)
        return result

    @staticmethod
    def evaluate(results: List[pd.DataFrame]):
        acc_list, f1_list, precision_list, recall_list = [], [], [], []
        for rep_result_df in results:
            rep_result_metrics = compute_metrics(rep_result_df['label'], rep_result_df['answer'])
            acc_list.append(rep_result_metrics['accuracy'])
            f1_list.append(rep_result_metrics['f1'])
            precision_list.append(rep_result_metrics['precision'])
            recall_list.append(rep_result_metrics['recall'])
        avg = (np.mean(acc_list), np.mean(f1_list), np.mean(precision_list), np.mean(recall_list))
        std = (np.std(acc_list), np.std(f1_list), np.std(precision_list), np.std(recall_list))
        return avg, std

    @staticmethod
    def _read_dataset(dataset: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset)
        return df[['label', 'text']]
