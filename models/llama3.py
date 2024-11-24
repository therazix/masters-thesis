import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import TrainingArguments
from trl import SFTTrainer

if os.getenv('IMPORT_FOR_LLM') == '1':
    from unsloth import is_bfloat16_supported, FastLanguageModel

from utils import get_child_logger
from . import HuggingFaceLLM, UnslothLLM


class Llama3(HuggingFaceLLM):
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(
            output_dir=output_dir,
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            dataset_path=dataset_path,
            template=template,
            hf_token=hf_token,
            logger=get_child_logger(__name__, logger))

    def _generate(self, query: str, examples: str):
        messages = self.format_prompts(query, examples)
        formated_messages = self.tokenizer.apply_chat_template(messages,
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        tokenized_messages = self.tokenizer(formated_messages, return_tensors='pt',
                                            padding=True).to(self.model.device)
        input_ids = tokenized_messages['input_ids']
        attention_mask = tokenized_messages['attention_mask']
        prompt_length = input_ids.size(-1)
        outputs = self.model.generate(input_ids,
                                      top_p=1.0,
                                      temperature=None,
                                      do_sample=False,
                                      attention_mask=attention_mask,
                                      max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    def test(self):
        result = []
        for rep, rep_data in self.dataset.groupby(level=0):
            examples = json.dumps(
                {row['label']: row['example_text'] for _, row in rep_data.iterrows()},
                ensure_ascii=False
            )
            data = rep_data.sample(self.num_of_authors)
            responses = []
            for _, row in data.iterrows():
                response_str = self._generate(row['query_text'], examples)
                response = self.parse_response(response_str)
                response['label'] = str(row['label'])
                response['rep'] = str(rep)
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        avg, std = self.evaluate(result)
        self.save_results('llama3', result, avg, std)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')

class Llama3FT(UnslothLLM):
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 model_name: str,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        os.environ['WANDB_MODE'] = 'disabled'  # disable wandb
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            chat_template='llama-3',
            dataset_path=dataset_path,
            template=template,
            hf_token=hf_token,
            logger=get_child_logger(__name__, logger))


    def finetune(self, dataset_name: str, repo_id: str, epochs: int = 6):
        dataset = self.load_finetuning_dataset(dataset_name)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            dataset_text_field='text',
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=20,
                num_train_epochs=epochs,
                eval_strategy='steps',
                save_steps=20,
                eval_steps=20,
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim='adamw_8bit',
                weight_decay=0.01,
                lr_scheduler_type='linear',
                seed=3407,
                output_dir='output'
            ),
        )

        trainer.train()

        trainer.model.push_to_hub(repo_id, token=self.hf_token)
        trainer.tokenizer.push_to_hub(repo_id, token=self.hf_token)

    def _generate(self, query: str, examples: str):
        messages = self.format_prompts(query, examples)
        formated_messages = self.tokenizer.apply_chat_template(messages,
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        tokenized_messages = self.tokenizer(formated_messages, return_tensors='pt',
                                            padding=True).to(self.model.device)
        input_ids = tokenized_messages['input_ids']
        attention_mask = tokenized_messages['attention_mask']
        prompt_length = input_ids.size(-1)
        outputs = self.model.generate(input_ids,
                                      attention_mask=attention_mask,
                                      max_new_tokens=self.max_new_tokens,
                                      use_cache=True)
        return self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    def test(self):
        FastLanguageModel.for_inference(self.model)

        result = []
        for rep, rep_data in self.dataset.groupby(level=0):
            examples = '\n'.join(f'Autor {row["label"]}: {row["example_text"]}' for _, row in rep_data.iterrows())
            data = rep_data.sample(self.num_of_authors)
            responses = []
            for _, row in data.iterrows():
                response_str = self._generate(row['query_text'], examples)
                response = self.parse_response(response_str)
                response['label'] = str(row['label'])
                response['rep'] = str(rep)
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        avg, std = self.evaluate(result)
        self.save_results('llama3_ft', result, avg, std)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
