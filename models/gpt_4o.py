import json
import logging
import os
from pathlib import Path
from typing import Optional, cast

import dotenv
import pandas as pd
import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from utils import get_child_logger
from . import BaseLLM

MODEL_NAME = 'gpt-4o-2024-08-06'

class GPTResponse(BaseModel):
    analysis: str
    answer: str

class GPT4o(BaseLLM):
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 template: str,
                 openai_api_key: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.client = self.get_client(openai_api_key)
        self.model_name = MODEL_NAME
        self.model_name_for_encoding = 'gpt-4o'
        super().__init__(
            output_dir=output_dir,
            dataset_path=dataset_path,
            template=template,
            max_tokens=128000,
            max_new_tokens=800,
            logger=get_child_logger(__name__, logger)
        )

    @staticmethod
    def get_client(openai_api_key: Optional[str]) -> OpenAI:
        if openai_api_key is None:
            dotenv.load_dotenv()
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key is None:
                raise ValueError('OpenAI API key is not set. Use OPENAI_API_KEY environment '
                                 'variable, or pass it as an argument.')
        return OpenAI(api_key=openai_api_key)

    def count_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name_for_encoding)
        return len(encoding.encode(text))

    def _generate(self, query: str, examples: str) -> GPTResponse | str | None:
        messages = self.format_prompts(query, examples)
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=GPTResponse,
                max_tokens=800,
                temperature=0.0,
            )
            response = completion.choices[0].message
            if response.parsed:
                return cast(GPTResponse, response.parsed)
            elif response.content:
                return response.content
            return None
        except Exception as err:
            self.logger.error(err)
            return str(err)

    @staticmethod
    def _parse_response(response: GPTResponse | str | None):
        if response is None:
            return {'analysis': 'error', 'answer': 'error'}
        if isinstance(response, str):
            return {'analysis': response, 'answer': 'error'}
        return {'analysis': response.analysis, 'answer': response.answer}

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
                gpt_response = self._generate(row['query_text'], examples)
                response = self._parse_response(gpt_response)
                response['label'] = str(row['label'])
                response['rep'] = str(rep)
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        self.save_results('gpt4o', result)
        avg, std = self.evaluate(result)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
