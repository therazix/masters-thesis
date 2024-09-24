import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from metrics import compute_metrics
from datasets import Dataset, load_dataset


def process_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df[['label', 'text']]


class XLMRoberta:
    def __init__(self,
                 training_set: Path,
                 testing_set: Path,
                 checkpoint_path: Path,
                 model_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path.resolve()
        self.model_path = (model_path or checkpoint_path).resolve()
        self.model_base = 'xlm-roberta-base'  # Huggingface model name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_base)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_df = process_dataset(training_set)
        self.test_df = process_dataset(testing_set)
        self.train_dataset = self._tokenize_data(Dataset.from_pandas(self.train_df, split='train'))
        self.test_dataset = self._tokenize_data(Dataset.from_pandas(self.test_df, split='test'))
        self._init_directories()
        self.trainer = None

    def _tokenize_data(self, dataset: Dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        return dataset.map(tokenize_function, batched=True)

    def _get_dataset_stats(self) -> Tuple[int, Dict[int, str], Dict[str, int]]:
        author_names = self.test_df['label'].unique()
        num_of_authors = len(author_names)
        id2label = {int(i): str(author_names[i]) for i in range(num_of_authors)}
        label2id = {str(author_names[i]): int(i) for i in range(num_of_authors)}
        return num_of_authors, id2label, label2id

    def _init_directories(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def _get_training_args(self, epochs: int):
        return TrainingArguments(
            output_dir=str(self.checkpoint_path),
            learning_rate=2e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=max(epochs, 1),
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=['tensorboard'],
        )

    def train(self, epochs: int, resume_training: bool = False):
        training_args = self._get_training_args(epochs)
        num_of_authors, id2label, label2id = self._get_dataset_stats()

        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.checkpoint_path) if resume_training else self.model_base,
            num_labels=num_of_authors,
            id2label=id2label,
            label2id=label2id
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

        # Train model
        self.trainer.train()

        # Save model
        self.tokenizer.save_pretrained(str(self.model_path / 'tokenizer'))
        self.trainer.save_model(str(self.model_path / 'model'))

    def predict(self, input_data: Dataset, tokenize: bool = True):
        if tokenize:
            input_data = self._tokenize_data(input_data)
        return self.trainer.predict(input_data)
