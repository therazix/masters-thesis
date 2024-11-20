import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, \
    AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import prompts
from metrics import compute_metrics
from utils import get_child_logger


def get_hf_token(hf_token: Optional[str] = None) -> str:
    if hf_token:
        return hf_token
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError('Hugging Face API token is missing. Please set HF_TOKEN '
                         'environment variable or pass it as an argument.')
    return token


class BaseLLM:
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 template: str,
                 max_tokens: int,
                 max_new_tokens: int,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or get_child_logger(__name__)
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template = self._load_template(template)
        self.template_name = template
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        df, num_of_authors = self._read_dataset(dataset_path)
        self.num_of_authors = num_of_authors
        self.max_tokens_per_text = self._get_max_tokens_per_text(num_of_authors)
        self.dataset = self._parse_dataset(df)

    @staticmethod
    def _load_template(template: str) -> List[Dict[str, str]]:
        if template.lower() == 'en':
            return prompts.prompts_en
        if template.lower() == 'cz':
            return prompts.prompts_cz
        if template.lower() == 'cz-1shot':
            return prompts.prompts_cz_1shot
        if template.lower() == 'cz-inference':
            return prompts.prompts_cz_inference
        raise ValueError('Invalid template. Available templates are: en, cz, cz-1shot, cz-inference')

    def format_prompts(self, query: str, examples: str):
        messages = copy.deepcopy(self.template)
        messages[-1]['content'] = messages[-1]['content'].format(query_text=query, example_text=examples)
        return messages

    def _parse_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df['query_text'] = df['query_text'].apply(lambda x: self._crop_if_needed(x))
        df['example_text'] = df['example_text'].apply(lambda x: self._crop_if_needed(x))
        return df

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def _is_token_count_valid(self, text: str) -> bool:
        return self.count_tokens(text) < self.max_tokens_per_text

    def _crop_if_needed(self, text: str) -> str:
        if self._is_token_count_valid(text):
            return text

        sentences = re.split(r'(?<=[.!?]) +', text)
        cropped_text = ''
        for sentence in sentences:
            if self.count_tokens(cropped_text + sentence) > self.max_tokens_per_text:
                break
            cropped_text += sentence + ' '
        self.logger.warning(f"Text was cropped to fit the token limit: '{text}'")
        return cropped_text.strip()

    def _get_max_tokens_per_text(self, num_of_texts: int) -> int:
        buffer = 50
        template_len = self.count_tokens(str(self.template))
        rest = self.max_tokens - self.max_new_tokens - template_len - buffer
        if num_of_texts > 0:
            return rest // (num_of_texts + 1)  # +1 for the query text
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
    def _read_dataset(dataset: Path) -> Tuple[pd.DataFrame, int]:
        df = pd.read_csv(dataset, index_col=0)
        num_of_authors = df['label'].groupby(level=0).nunique().unique()[0]
        return df, num_of_authors

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

    def save_results(self, model_name: str, results: List[pd.DataFrame]):
        reps = self.dataset.index.get_level_values(0).nunique()
        filename = f'{model_name}_{self.num_of_authors}authors_{reps}reps_{self.template_name}.csv'
        results_df = pd.concat(results, ignore_index=True)
        results_df.to_csv(self.output_dir / filename)


class HuggingFaceLLM(BaseLLM):
    def __init__(self,
                 output_dir: Path,
                 model_name: str,
                 dataset_path: Path,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.hf_token = get_hf_token(hf_token)
        self._login(self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.float16,
                                                          attn_implementation='flash_attention_2',
                                                          device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(
            output_dir=output_dir,
            dataset_path=dataset_path,
            template=template,
            max_tokens=self.model.config.max_position_embeddings,
            max_new_tokens=800,
            logger=logger or get_child_logger(__name__)
        )

    @staticmethod
    def _login(token: str):
        login(token)

    def count_tokens(self, text: str) -> int:
        encoding = self.tokenizer(text)
        return len(encoding.input_ids)


class UnslothLLM(BaseLLM):
    def __init__(self,
                 output_dir: Path,
                 model_name: str,
                 chat_template: str,
                 dataset_path: Path,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.hf_token = get_hf_token(hf_token)
        self._login(self.hf_token)

        self.max_seq_length = 1024

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # None = autodetect
            load_in_4bit=True,
            token=self.hf_token,
        )

        self.model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=37,
            use_rslora=False,
            loftq_config=None,
        )
        self.tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

        super().__init__(
            output_dir=output_dir,
            dataset_path=dataset_path,
            template=template,
            max_tokens=self.model.config.max_position_embeddings,
            max_new_tokens=1000,
            logger=logger or get_child_logger(__name__)
        )

    @staticmethod
    def _login(token: str):
        login(token)

    def count_tokens(self, text: str) -> int:
        encoding = self.tokenizer(text)
        return len(encoding.input_ids)

    @staticmethod
    def _format_finetuning_template(row):
        messages = copy.deepcopy(prompts.prompts_cz_finetuning)
        query_text = row["query_text"]
        examples = json.loads(row["example_text"])
        example_text = '\n'.join(f'Autor {key}: {value}' for key, value in examples.items())
        response = row["response"]
        label = row["label"]

        messages[1]['content'] = messages[1]['content'].format(query_text=query_text, example_text=example_text)
        messages[2]['content'] = messages[2]['content'].format(response=response, label=label)
        return messages

    def _format_finetuning_prompts(self, row):
        messages = self._format_finetuning_template(row)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        row["text"] = text
        return row

    def load_finetuning_dataset(self, dataset_name: str) -> DatasetDict:
        if Path(dataset_name).exists():
            dataset = load_dataset('csv', data_files=dataset_name, split='all')
        else:
            dataset = load_dataset(dataset_name, split='all')

        dataset = dataset.map(self._format_finetuning_prompts)
        return dataset.train_test_split(test_size=0.1)
