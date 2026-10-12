# Banking Customer Intent Classification using DistilBERT

An end-to-end text classification project that fine-tunes DistilBERT to route banking customer messages to one of the 77 BANKING77 intents. The first-run defaults use a reproducible, stratified 500-message training subset and 100-message test subset so the complete workflow can be checked with modest compute.

## Dataset

[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) is a public banking customer-support intent dataset with 13,083 English messages across 77 fine-grained intents. The Hugging Face Hub entry uses a legacy loading script, so this project uses Hugging Face `datasets`' CSV builder with the canonical train and test files linked by that entry. The files are cached automatically; no manual dataset setup is needed.

## Technology

Python, PyTorch, Hugging Face Transformers and Datasets, DistilBERT (`distilbert-base-uncased`), scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, and Streamlit.

## Project Structure

```text
.
├── data/                 # Optional local data; BANKING77 is cached by Datasets
├── models/               # Trained model, tokenizer, label map, and evaluation output
├── src/
│   ├── data_loader.py    # Dataset download and stratified smoke subsets
│   ├── preprocess.py     # Fixed-length DistilBERT tokenization
│   ├── train.py          # Fine-tuning and model artifact saving
│   ├── evaluate.py       # Metrics, classification report, confusion matrix
│   ├── predict.py        # Reusable and command-line prediction
│   └── utils.py          # Seeds, paths, and label mapping
├── app.py                # Streamlit prediction interface
├── requirements.txt
└── runtime.txt
```

## Installation

Python 3.11 or 3.12 is recommended. In GitHub Codespaces, open a terminal at the repository root and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch uses CUDA automatically when available and otherwise runs on CPU. The first run downloads the dataset and pretrained model weights from Hugging Face.

## Train: Smoke Run

The defaults are one epoch, batch size 16, learning rate `2e-5`, max sequence length 128, 500 stratified training examples, and 100 stratified test examples:

```bash
python -m src.train
```

The model and tokenizer are saved in `models/distilbert-banking77/`; `models/label_mapping.json` preserves the dataset's 77 label IDs. Training artifacts are intentionally excluded from Git because model weights are large.

## Evaluate

Run evaluation against the same 100-example smoke test subset:

```bash
python -m src.evaluate
```

The command prints accuracy, weighted precision, weighted recall, and weighted F1. It writes `models/evaluation_metrics.json`, `models/classification_report.txt` (per-intent precision, recall, F1, and support), and `models/confusion_matrix.png`.

## Predict

```bash
python -m src.predict --text "I tried to withdraw money from an ATM but no cash was dispensed."
```

The command prints the predicted intent and its softmax confidence. The same `predict_intent` function is used by the web app.

## Streamlit

Start the app after training:

```bash
streamlit run app.py
```

Codespaces forwards the Streamlit port (8501); open the forwarded URL to enter a message and see its predicted intent and confidence.

## Full BANKING77 Training

The 500/100 defaults are only for first-run validation. Pass zero for both sample limits to use every example in the published train and test splits:

```bash
python -m src.train --train-samples 0 --test-samples 0 --epochs 3
python -m src.evaluate --test-samples 0
```

Full training is substantially slower and uses more memory, especially on CPU. Increase epochs or tune batch size based on available hardware. You can also set `--batch-size`, `--learning-rate`, `--max-length`, and `--seed` explicitly.

## Expected Output

The smoke-test evaluation reports four scalar metrics and creates the classification report and confusion-matrix image. Accuracy and F1 depend on the initialized pretrained weights, sample subset, and training environment; the small smoke run is a pipeline check, not a benchmark. BANKING77 does not contain an intent named `cash_withdrawal`; similar canonical labels include `declined_cash_withdrawal` and `cash_withdrawal_not_recognised`. The smoke model's prediction is not guaranteed to match either label.

## Future Improvements

Use a validation split for early stopping and hyperparameter selection, compare macro and weighted metrics, add confidence calibration, persist experiment tracking, and serve the model behind an authenticated API for production use.

## GitHub

After reviewing the generated source and choosing whether to include any local evaluation artifacts, publish the source with:

```bash
git add .
git commit -m "Complete Banking77 DistilBERT Intent Classification"
git push
```