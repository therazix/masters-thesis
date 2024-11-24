import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import get_child_logger
from . import HuggingFaceLLM


class Mistral(HuggingFaceLLM):
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(
            output_dir=output_dir,
            model_name='mistralai/Mistral-7B-Instruct-v0.3',
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
        self.save_results('mistral', result, avg, std)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
