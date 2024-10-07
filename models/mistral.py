import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts
from metrics import compute_metrics
from utils import get_child_logger

MAX_TOKENS = 32000
MAX_NEW_TOKENS = 400
MIN_TOKENS_PER_TEXT = 56


def _login(hf_token: Optional[str] = None):
    load_dotenv()
    token = hf_token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError('Hugging Face API token is missing. Please set HF_TOKEN '
                         'environment variable or pass it as an argument.')
    login(token=token)


def _extract_samples(df: pd.DataFrame) -> pd.DataFrame:
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


def _evaluate(results: List[pd.DataFrame]):
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


def _read_dataset(dataset: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    return df[['label', 'text']]


class Mistral:
    def __init__(self,
                 dataset: Path,
                 lang: str,
                 crop: bool,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = get_child_logger(__name__, logger)
        if lang.lower() not in ['en', 'cz']:
            raise ValueError("Language must be either 'en' or 'cz'")
        _login(hf_token)
        self.model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='auto')
        self.logger.info(f'MAX TOKENS: {self.model.config.max_position_embeddings}')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        df = _read_dataset(dataset)
        self.num_of_authors = len(df['label'].unique())
        self.template = prompts.mistral_prompts_en if lang == 'en' else prompts.mistral_prompts_cz
        self.max_tokens_per_text = self._get_max_tokens_per_text(self.template, self.num_of_authors)
        self.dataset = self._parse_dataset(df, crop)
        self.logger.info(f'MAX TOKENS PER TEXT: {self.max_tokens_per_text}')

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
        return MIN_TOKENS_PER_TEXT < count < self.max_tokens_per_text

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
        rest = MAX_TOKENS - MAX_NEW_TOKENS - template_len - buffer
        if num_of_examples > 0:
            return rest // num_of_examples
        return rest

    def _generate(self, query: str, examples: str):
        messages = [
            {
                'role': 'system',
                'content': self.template['system']
            },
            {
                'role': 'user',
                'content': self.template['user'].format(query=query, examples=examples)
            }
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors='pt').to('cuda')
        outputs = self.model.generate(model_inputs,
                                      top_p=1.0,
                                      max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0])

    def _parse_response(self, text: str):
        try:
            cleaned = text.split('[/INST]', maxsplit=1)[1].split('</s>', maxsplit=1)[0].strip()
            response = json.loads(cleaned, strict=False)
        except (json.JSONDecodeError, IndexError) as err:
            self.logger.info('Error while decoding response: ' + str(err))
            response = json.loads('{}')
            response['analysis'] = text
            response['answer'] = 'error'
        return response

    def test(self, reps: int):
        result = []
        for _ in range(reps):
            responses = []
            samples = _extract_samples(self.dataset)

            examples = json.dumps(
                {row['label']: row['example_text'] for _, row in samples.iterrows()},
                ensure_ascii=False
            )

            # Iterate over samples so that label and query_text are accessible
            for _, row in samples.iterrows():
                response_str = self._generate(row['query_text'], examples)
                response = self._parse_response(response_str)
                response['label'] = str(row['label'])
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        avg, std = _evaluate(result)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
