import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

import prompts
from utils import get_child_logger
from . import LLM


class Llama3(LLM):
    def __init__(self,
                 dataset_path: Path,
                 lang: str,
                 crop: bool,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        if lang.lower() not in ['en', 'cz']:
            raise ValueError("Language must be either 'en' or 'cz'")

        super().__init__(
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            dataset_path=dataset_path,
            crop=crop,
            template=prompts.prompts_en if lang == 'en' else prompts.prompts_cz,
            hf_token=hf_token,
            logger=get_child_logger(__name__, logger))

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
        formated_messages = self.tokenizer.apply_chat_template(messages,
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        tokenized_messages = self.tokenizer(formated_messages, return_tensors='pt',
                                            padding=True).to(self.model.device)
        prompt_length = tokenized_messages['input_ids'].size(-1)
        outputs = self.model.generate(tokenized_messages['input_ids'],
                                      top_p=1.0,
                                      temperature=None,
                                      do_sample=False,
                                      attention_mask=tokenized_messages['attention_mask'],
                                      max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    def _parse_response(self, text: str):
        try:
            json_start = text.index('{')
            json_end = len(text) - text[::-1].index('}')
            text = text[json_start:json_end]
            response = json.loads(text, strict=False)
        except (json.JSONDecodeError, IndexError, ValueError) as err:
            matches = re.findall(r'["\']?answer["\']?: ["\']?(\d+)["\']?', text)
            response = json.loads('{}')
            response['analysis'] = text
            if matches:
                response['answer'] = matches[-1]
            else:
                response['answer'] = 'error'
                self.logger.error(f'Failed to parse response: {text}')
        return response

    def test(self, reps: int):
        result = []
        for _ in range(reps):
            responses = []
            samples = self.extract_samples(self.dataset)

            examples = json.dumps(
                {row['label']: row['example_text'] for _, row in samples.iterrows()},
                ensure_ascii=False
            )

            for _, row in samples.iterrows():
                response_str = self._generate(row['query_text'], examples)
                response = self._parse_response(response_str)
                response['label'] = str(row['label'])
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        avg, std = self.evaluate(result)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
