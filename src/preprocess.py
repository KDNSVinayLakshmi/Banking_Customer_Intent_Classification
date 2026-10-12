"""DistilBERT tokenization helpers shared by training and evaluation."""

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


MAX_LENGTH = 128


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """Tokenize messages with the project's fixed padding and truncation."""
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(
            batch["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    return tokenized
