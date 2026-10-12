"""Fine-tune DistilBERT on BANKING77 and save a reusable model bundle."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import load_banking77_data
from src.preprocess import MAX_LENGTH, tokenize_dataset
from src.utils import DEFAULT_MODELS_DIR, PROJECT_ROOT, save_label_mapping, set_seed


MODEL_NAME = "distilbert-base-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir = args.output_dir.resolve()
    model_dir = args.output_dir / "distilbert-banking77"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, _, label_names = load_banking77_data(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    print(f"Training samples: {len(train_data)} | intents: {len(label_names)}")
    print(f"Device: {device} | model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label={index: name for index, name in enumerate(label_names)},
        label2id={name: index for index, name in enumerate(label_names)},
    ).to(device)
    tokenized_train = tokenize_dataset(train_data, tokenizer, args.max_length)
    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Each batch updates the classifier and DistilBERT encoder together.
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch["labels"] = batch.pop("label")
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f}"
                )
        print(f"Epoch {epoch + 1} mean loss: {total_loss / len(train_loader):.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    mapping_path = save_label_mapping(label_names, args.output_dir)
    config = {
        "base_model": MODEL_NAME,
        "train_samples": args.train_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Model and tokenizer saved to: {model_dir}")
    print(f"Label mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
