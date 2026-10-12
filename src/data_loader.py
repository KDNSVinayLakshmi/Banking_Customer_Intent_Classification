"""Load BANKING77 and make deterministic, label-stratified smoke-test splits."""

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset


DATASET_FILES = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def _stratified_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    """Return an exact-size subset with examples spread across all intents."""
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset

    random_generator = np.random.default_rng(seed)
    label_ids = np.asarray(dataset["label"])
    class_indices = [
        random_generator.permutation(np.flatnonzero(label_ids == label_id)).tolist()
        for label_id in range(dataset.features["label"].num_classes)
    ]
    selected_indices: list[int] = []
    # Round-robin selection gives each intent a training example before any
    # intent receives a second one, while still respecting class sizes.
    while len(selected_indices) < limit:
        added_index = False
        for indices in class_indices:
            if indices and len(selected_indices) < limit:
                selected_indices.append(indices.pop())
                added_index = True
        if not added_index:
            break

    random_generator.shuffle(selected_indices)
    return dataset.select(selected_indices)


def load_banking77_data(
    train_samples: int | None = 500,
    test_samples: int | None = 100,
    seed: int = 42,
) -> tuple[Dataset, Dataset, list[str]]:
    """Download/cache BANKING77 and return train, test, and ordered labels."""
    # The BANKING77 Hub entry is a legacy script; read its canonical CSV files
    # through Datasets' maintained CSV builder instead of executing that script.
    dataset: DatasetDict = load_dataset("csv", data_files=DATASET_FILES)
    label_names = sorted(dataset["train"].unique("category"))
    label_to_id = {name: index for index, name in enumerate(label_names)}

    def encode_labels(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return {"label": [label_to_id[name] for name in batch["category"]]}

    dataset = DatasetDict(
        {
            split: split_data.map(encode_labels, batched=True, remove_columns=["category"])
            for split, split_data in dataset.items()
        }
    )
    dataset = DatasetDict(
        {
            split: split_data.cast_column("label", ClassLabel(names=label_names))
            for split, split_data in dataset.items()
        }
    )
    train_data = _stratified_subset(dataset["train"], train_samples, seed)
    test_data = _stratified_subset(dataset["test"], test_samples, seed + 1)
    return train_data, test_data, label_names
