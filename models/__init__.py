import copy
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

import prompts
from metrics import compute_metrics
from utils import get_child_logger


class BaseLLM:
    def __init__(self,
                 dataset_path: Path,
                 template: str,
                 crop: bool,
                 min_tokens_per_text: int,
                 max_tokens: int,
                 max_new_tokens: int,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or get_child_logger(__name__)
        self.template = self._load_template(template)
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.min_tokens_per_text = min_tokens_per_text
        df = self._read_dataset(dataset_path)
        num_of_authors = len(df['label'].unique())
        self.max_tokens_per_text = self._get_max_tokens_per_text(num_of_authors)
        self.dataset = self._parse_dataset(df, crop)

    @staticmethod
    def _load_template(template: str) -> List[Dict[str, str]]:
        if template.lower() == 'en':
            return prompts.prompts_en
        if template.lower() == 'cz':
            return prompts.prompts_cz
        if template.lower() == 'cz-1shot':
            return prompts.prompts_cz_1shot
        raise ValueError('Invalid template. Available templates are: en, cz, cz-1shot')

    def format_prompts(self, query: str, examples: str):
        messages = copy.deepcopy(self.template)
        messages[-1]['content'] = messages[-1]['content'].format(query=query, examples=examples)
        return messages

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

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def _is_token_count_valid(self, text: str) -> bool:
        count = self.count_tokens(text)
        return self.min_tokens_per_text < count < self.max_tokens_per_text

    def _crop_if_needed(self, text: str) -> str:
        if self._is_token_count_valid(text):
            return text

        sentences = re.split(r'(?<=[.!?]) +', text)
        cropped_text = ''
        for sentence in sentences:
            if self.count_tokens(cropped_text + sentence) > self.max_tokens_per_text:
                break
            cropped_text += sentence + ' '
        return cropped_text.strip()

    def _get_max_tokens_per_text(self, num_of_examples: int) -> int:
        buffer = 50
        template_len = self.count_tokens(str(self.template))
        rest = self.max_tokens - self.max_new_tokens - template_len - buffer
        if num_of_examples > 0:
            return rest // num_of_examples
        return rest

    @staticmethod
    def extract_samples(df: pd.DataFrame) -> pd.DataFrame:
        author_names = df['label'].unique().tolist()
        result = pd.DataFrame(columns=['label', 'query_text', 'example_text'])
        for author in author_names:
            # Get 2 random texts from author
            try:
                text_1, text_2 = df[df['label'] == author]['text'].sample(2)
            except ValueError as err:
                raise ValueError(f'Not enough samples for author {author}.') from err
            author_df = pd.DataFrame([[author, text_1, text_2]], columns=result.columns)
            result = pd.concat([result, author_df], ignore_index=True)
        result = result.sort_values(by=['label'])
        result = result.reset_index(drop=True)
        return result

    @staticmethod
    def _read_dataset(dataset: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset)
        return df[['label', 'text']]

    @staticmethod
    def evaluate(results: List[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float]]:
        acc_list, f1_list, precision_list, recall_list = [], [], [], []
        for rep_result_df in results:
            rep_result_metrics = compute_metrics(rep_result_df['label'], rep_result_df['answer'])
            acc_list.append(rep_result_metrics['accuracy'])
            f1_list.append(rep_result_metrics['f1'])
            precision_list.append(rep_result_metrics['precision'])
            recall_list.append(rep_result_metrics['recall'])
        avg = {
            'accuracy': np.mean(acc_list),
            'f1': np.mean(f1_list),
            'precision': np.mean(precision_list),
            'recall': np.mean(recall_list)
        }
        std = {
            'accuracy': np.std(acc_list),
            'f1': np.std(f1_list),
            'precision': np.std(precision_list),
            'recall': np.std(recall_list)
        }
        return avg, std



class HuggingFaceLLM(BaseLLM):
    def __init__(self,
                 model_name: str,
                 dataset_path: Path,
                 crop: bool,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self._login(hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.float16,
                                                          attn_implementation='flash_attention_2',
                                                          device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(
            dataset_path=dataset_path,
            template=template,
            crop=crop,
            min_tokens_per_text=56,
            max_tokens=self.model.config.max_position_embeddings,
            max_new_tokens=800,
            logger=logger or get_child_logger(__name__)
        )

    @staticmethod
    def _login(hf_token: Optional[str] = None):
        load_dotenv()
        token = hf_token or os.getenv('HF_TOKEN')
        if not token:
            raise ValueError('Hugging Face API token is missing. Please set HF_TOKEN '
                             'environment variable or pass it as an argument.')
        login(token=token)

    def count_tokens(self, text: str) -> int:
        encoding = self.tokenizer(text)
        return len(encoding.input_ids)

