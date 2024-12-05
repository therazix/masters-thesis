import logging
from pathlib import Path
from typing import Optional

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
            short_model_name='llama',
            chat_template='llama-3',
            dataset_path=dataset_path,
            template=template,
            hf_token=hf_token,
            logger=get_child_logger(__name__, logger))
