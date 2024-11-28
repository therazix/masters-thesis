import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from unsloth import FastLanguageModel

from utils import get_child_logger
from . import UnslothLLM


class Llama3(UnslothLLM):
    def __init__(self,
                 output_dir: Path,
                 dataset_path: Path,
                 model_name: str,
                 template: str,
                 hf_token: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            chat_template='llama-3',
            dataset_path=dataset_path,
            template=template,
            hf_token=hf_token,
            logger=get_child_logger(__name__, logger))


    def test(self):
        FastLanguageModel.for_inference(self.model)

        result = []
        for rep, rep_data in self.dataset.groupby(level=0):
            examples = '\n'.join(f'Autor {row["label"]}: {row["example_text"]}' for _, row in rep_data.iterrows())
            data = rep_data.sample(self.num_of_authors)
            responses = []
            for _, row in data.iterrows():
                response_str = self.generate(row['query_text'], examples)
                response = self.parse_response(response_str)
                response['label'] = str(row['label'])
                response['rep'] = str(rep)
                responses.append(response)
                self.logger.info(str(response))
            responses_df = pd.DataFrame(responses)
            responses_df[['label', 'answer']] = responses_df[['label', 'answer']].astype(str)
            result.append(responses_df)

        avg, std = self.evaluate(result)
        self.save_results('llama3.1', result, avg, std)
        self.logger.info(f'Average: {avg}')
        self.logger.info(f'Standard deviation: {std}')
