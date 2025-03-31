import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DataCollatorWithPadding, Trainer, TrainingArguments, EvalPrediction

import metrics
from utils import get_child_logger


def process_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df[['label', 'text']]

def compute_metrics_hf(eval_pred: EvalPrediction):
    """
    Compute metrics for Huggingface Trainer
    """
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions.argmax(-1)
    return metrics.compute_metrics(labels, predictions)

def get_dataset_info(df: pd.DataFrame) -> Tuple[int, Dict[int, str], Dict[str, int]]:
    author_names = sorted(df['label'].unique())
    num_of_authors = len(author_names)
    id2label = {int(i): str(author_names[i]) for i in range(num_of_authors)}
    label2id = {str(author_names[i]): int(i) for i in range(num_of_authors)}
    return num_of_authors, id2label, label2id


class XLMRoberta:
    def __init__(self,
                 model: str,
                 output_dir: Path,
                 training_set: Optional[Path] = None,
                 validation_set: Optional[Path] = None,
                 testing_set: Optional[Path] = None,
                 epochs: Optional[int] = None,
                 hub_model_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None
                 ):
        self.logger = get_child_logger(__name__, logger)
        self.model_name_or_path = model
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = max(epochs, 1) if epochs else 1
        self.hub_model_id = hub_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if not any([training_set, validation_set, testing_set]):
            raise ValueError('At least one dataset should be provided')

        df = None
        if training_set:
            df = process_dataset(training_set)
            self.train_dataset = self._tokenize_data(Dataset.from_pandas(df, split='train'))
        if validation_set:
            df = process_dataset(validation_set)
            self.val_dataset = self._tokenize_data(Dataset.from_pandas(df, split='validation'))
        if testing_set:
            df = process_dataset(testing_set)
            self.test_dataset = self._tokenize_data(Dataset.from_pandas(df, split='test'))

        if df is None:
            raise ValueError('No valid dataset provided')

        num_of_authors, id2label, label2id = get_dataset_info(df)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            push_to_hub=bool(self.hub_model_id),
            report_to='none',
            hub_model_id=self.hub_model_id
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=num_of_authors,
            id2label=id2label,
            label2id=label2id
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics_hf,
        )

    def _tokenize_data(self, dataset: Dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        return dataset.map(tokenize_function, batched=True)

    def train(self):
        if self.train_dataset is None:
            raise ValueError('Training set is not provided')
        if self.val_dataset is None:
            raise ValueError('Validation set is not provided')

        start_time = time.time()
        self.trainer.train()
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.info(f'Training completed in {time_elapsed}')

        if self.hub_model_id:
            self.trainer.push_to_hub()

        self.logger.info(f'Validation set results: {self.trainer.evaluate()}')
        if self.test_dataset:
            self.logger.info(f'Test set results: {self.trainer.evaluate(self.test_dataset)}')

    def test(self):
        if self.test_dataset is None:
            raise ValueError('Testing set is not provided')
        self.logger.info(f'Test set results: {self.trainer.evaluate(self.test_dataset)}')

    def predict(self, input_data: Dataset, tokenize: bool = True):
        if tokenize:
            input_data = self._tokenize_data(input_data)
        return self.trainer.predict(input_data)
