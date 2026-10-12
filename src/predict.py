"""Predict a BANKING77 intent and confidence for a custom customer message."""

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import MAX_LENGTH
from src.utils import DEFAULT_MODEL_DIR, DEFAULT_MODELS_DIR, load_label_names


def load_model(model_dir: Path = DEFAULT_MODEL_DIR) -> tuple[Any, Any, torch.device, list[str]]:
    """Load model, tokenizer, device, and labels once for prediction."""
    model_dir = model_dir.resolve()
    mapping_path = model_dir.parent / "label_mapping.json"
    label_names = load_label_names(mapping_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, device, label_names


def predict_intent(
    text: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    loaded_model: tuple[Any, Any, torch.device, list[str]] | None = None,
    max_length: int = MAX_LENGTH,
) -> dict[str, float | str]:
    """Return the highest-scoring intent and its softmax confidence."""
    if not text.strip():
        raise ValueError("Message text must not be empty.")
    model, tokenizer, device, label_names = loaded_model or load_model(model_dir)
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    return {
        "intent": label_names[label_id],
        "confidence": float(probabilities[label_id].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Customer message to classify")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    result = predict_intent(args.text, args.model_dir)
    print(f"Predicted intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
