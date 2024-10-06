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

MAX_TOKENS_MISTRAL = 32000


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


class Mistral:
    def __init__(self, dataset: Path, crop: bool, hf_token: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.logger = get_child_logger(__name__, logger)
        _login(hf_token)
        self.model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset = self._parse_dataset(dataset.resolve(), crop)
        self.num_of_authors = len(self.dataset['label'].unique())
        self.template = prompts.mistral_prompts_en
        self.max_tokens_per_text = self._get_max_tokens_per_text(self.template, self.num_of_authors)
        self.logger.info(f'MAX TOKENS: {self.max_tokens_per_text}')

    def _parse_dataset(self, dataset: Path, crop: bool = False) -> pd.DataFrame:
        df = pd.read_csv(dataset)
        df = df[['label', 'text']]
        if crop:
            result = df[df['text'].parallel_apply(lambda x: self._crop_if_needed(x))]
        else:
            result = df[df['text'].parallel_apply(lambda x: self._is_token_count_valid(x))]

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

    def _is_token_count_valid(self, text: str, min: int = 56, max: int = 512) -> bool:
        count = self._count_tokens(text)
        return count > min and count < max

    def _crop_if_needed(self, text: str) -> str:
        max = self._get_max_tokens_per_text()
        if self._is_token_count_valid(text, max=max):
            return text

        sentences = re.split(r'(?<=[.!?]) +', text)
        cropped_text = ''
        for sentence in sentences:
            if self._count_tokens(cropped_text + sentence) > max:
                break
            cropped_text += sentence + ' '
        return cropped_text.strip()

    def _get_max_tokens_per_text(self, template: Dict[str, str], num_of_examples: int) -> int:
        buffer = 50
        template_len = self._count_tokens(str(template))
        rest = MAX_TOKENS_MISTRAL - template_len - buffer
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
        generated_ids = self.model.generate(model_inputs, top_p=1.0, max_new_tokens=4096,
                                            do_sample=False,
                                            pad_token_id=self.tokenizer.eos_token_id)
        response_str = generated_ids[0].outputs[0].text.strip()
        return response_str

    def _parse_response(self, text: str):
        try:
            response = json.loads(text, strict=False)
        except json.JSONDecodeError:
            self.logger.info('Error while decoding response.')
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
                response['label'] = row['label']
                responses.append(response)
                self.logger.info(str(response))

            result.append(pd.DataFrame(responses))

        avg, std = _evaluate(result)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
