"""Shared paths, reproducibility helpers, and label-map serialization."""

import json
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_DIR = DEFAULT_MODELS_DIR / "distilbert-banking77"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable sampling and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_label_mapping(label_names: list[str], output_dir: Path) -> Path:
    """Persist the dataset's stable label IDs for training and inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "id2label": {str(index): name for index, name in enumerate(label_names)},
        "label2id": {name: index for index, name in enumerate(label_names)},
    }
    mapping_path = output_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return mapping_path


def load_label_names(mapping_path: Path) -> list[str]:
    """Load label names in numeric ID order from the saved mapping."""
    with mapping_path.open(encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    id_to_label = mapping["id2label"]
    return [id_to_label[str(index)] for index in range(len(id_to_label))]
