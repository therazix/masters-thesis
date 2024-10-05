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


class XLMRoberta:
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        self.logger = get_child_logger(__name__, logger)
        self.output_dir = output_dir.resolve()
        self.model_base = 'xlm-roberta-base'  # Huggingface model name

        self.trainer = None
        self.train_df, self.train_dataset = None, None
        self.val_df, self.val_dataset = None, None
        self.test_df, self.test_dataset = None, None
        self._init_directories()

    def _init_train(self,
                    training_set: Path,
                    validation_set: Path,
                    testing_set: Optional[Path] = None,
                    checkpoint: Optional[Path] = None):
        self.model_name_or_path = str(checkpoint.resolve()) if checkpoint else self.model_base
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_df = process_dataset(training_set)
        self.train_dataset = self._tokenize_data(
            Dataset.from_pandas(self.train_df, split='train'))

        self.val_df = process_dataset(validation_set) if validation_set else None
        self.val_dataset = self._tokenize_data(
            Dataset.from_pandas(self.val_df, split='validation'))

        if testing_set:
            self.test_df = process_dataset(testing_set) if testing_set else None
            self.test_dataset = self._tokenize_data(
                Dataset.from_pandas(self.test_df, split='test'))

    def _init_test(self, model: Path, testing_set: Path):
        self.model_name_or_path = str(model.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.test_df = process_dataset(testing_set)
        self.test_dataset = self._tokenize_data(
            Dataset.from_pandas(self.test_df, split='test'))

    @classmethod
    def for_training(cls,
                     output_dir: Path,
                     training_set: Path,
                     validation_set: Path,
                     testing_set: Optional[Path] = None,
                     checkpoint: Optional[Path] = None,
                     logger: Optional[logging.Logger] = None):
        xlm_roberta = cls(output_dir, logger)
        xlm_roberta._init_train(training_set, validation_set, testing_set, checkpoint)
        return xlm_roberta

    @classmethod
    def for_testing(cls, output_dir: Path, model: Path, testing_set: Path, logger: Optional[logging.Logger] = None):
        xlm_roberta = cls(output_dir, logger)
        xlm_roberta._init_test(model, testing_set)
        xlm_roberta._init_trainer(1)
        return xlm_roberta

    def _tokenize_data(self, dataset: Dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        return dataset.map(tokenize_function, batched=True)

    def _get_dataset_info(self) -> Tuple[int, Dict[int, str], Dict[str, int]]:
        author_names = sorted(self.test_df['label'].unique())
        num_of_authors = len(author_names)
        id2label = {int(i): str(author_names[i]) for i in range(num_of_authors)}
        label2id = {str(author_names[i]): int(i) for i in range(num_of_authors)}
        return num_of_authors, id2label, label2id

    def _init_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_trainer(self, epochs: int):
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=max(epochs, 1),
            weight_decay=0.01,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            push_to_hub=False,
            report_to='none',
        )
        num_of_authors, id2label, label2id = self._get_dataset_info()

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
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics_hf,
        )

    def train(self, epochs: int):
        if self.train_dataset is None:
            raise ValueError('Training set is not provided')

        self._init_trainer(epochs)

        # Train model
        start_time = time.time()
        self.trainer.train()
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.info(f'Training completed in {time_elapsed}')

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
