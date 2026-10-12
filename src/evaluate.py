"""Evaluate a saved BANKING77 classifier and write metrics and a confusion matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import load_banking77_data
from src.preprocess import MAX_LENGTH, tokenize_dataset
from src.utils import DEFAULT_MODEL_DIR, DEFAULT_MODELS_DIR, load_label_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping_path = args.output_dir / "label_mapping.json"
    label_names = load_label_names(mapping_path)

    _, test_data, _ = load_banking77_data(
        train_samples=0,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    test_loader = DataLoader(
        tokenize_dataset(test_data, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    actual: list[int] = []
    predicted: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            labels = batch.pop("label")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())

    label_ids = np.arange(len(label_names))
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision_weighted": precision_score(
            actual, predicted, labels=label_ids, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            actual, predicted, labels=label_ids, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            actual, predicted, labels=label_ids, average="weighted", zero_division=0
        ),
        "test_samples": len(actual),
    }
    report = classification_report(
        actual,
        predicted,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
        digits=4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(actual, predicted, labels=label_ids)
    figure, axis = plt.subplots(figsize=(22, 20))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False)
    axis.set_title("BANKING77 Intent Classification Confusion Matrix")
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    print(f"Evaluation samples: {metrics['test_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Classification report: {args.output_dir / 'classification_report.txt'}")
    print(f"Confusion matrix: {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
