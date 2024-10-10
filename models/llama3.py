import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import prompts
from utils import get_child_logger
from . import LLM
import transformers

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
        tokenized_messages = self.tokenizer.apply_chat_template(messages,
                                                           return_tensors='pt',
                                                           add_generation_prompt=True).to('cuda')
        prompt_length = tokenized_messages.size(-1)
        outputs = self.model.generate(tokenized_messages,
                                      top_p=1.0,
                                      do_sample=False,
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
            self.logger.info('Error while decoding response: ' + str(err))
            response = json.loads('{}')
            response['analysis'] = text
            response['answer'] = 'error'
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
