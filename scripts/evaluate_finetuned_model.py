#!/usr/bin/env python3
"""Evaluate a fine-tuned encoder model on generated JSONL examples."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.training.data import get_error_labels, get_serialized_input, load_jsonl
from semantic_api_diagnosis.training.labels import ERROR_LABELS, labels_to_multihot, multihot_to_labels
from semantic_api_diagnosis.training.metrics import compute_multilabel_metrics
from semantic_api_diagnosis.serialization.jsonl import write_jsonl


def main() -> None:
    parser = ArgumentParser(description="Evaluate a fine-tuned model on JSONL examples.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    _validate_model_dir(model_dir)
    examples = load_jsonl(args.input_jsonl)
    predictions = predict_examples(
        model_dir=model_dir,
        examples=examples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    output_path = Path(args.output_jsonl)
    write_jsonl(predictions, output_path)
    if _all_have_gold(examples):
        metrics = compute_metrics_for_prediction_rows(predictions, threshold=args.threshold)
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote predictions to {output_path}")


def predict_examples(
    model_dir: Path,
    examples: list[dict],
    threshold: float,
    batch_size: int,
    max_length: int,
) -> list[dict]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Prediction requires torch and transformers. Install with: "
            'python3 -m pip install -e ".[train]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            texts = [get_serialized_input(example) for example in batch]
            encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            outputs = model(**encoded)
            probabilities = torch.sigmoid(outputs.logits).cpu().tolist()
            for example, scores in zip(batch, probabilities):
                rows.append(format_prediction_row(example, scores, threshold))
    return rows


def format_prediction_row(example: dict, scores: list[float], threshold: float) -> dict:
    if len(scores) != len(ERROR_LABELS):
        raise ValueError(f"Expected {len(ERROR_LABELS)} scores, got {len(scores)}")
    gold_labels = get_error_labels(example) if _has_gold(example) else []
    return {
        "id": example.get("id"),
        "gold_error_labels": gold_labels,
        "predicted_error_labels": multihot_to_labels(scores, threshold=threshold),
        "scores": {label: float(score) for label, score in zip(ERROR_LABELS, scores)},
    }


def compute_metrics_for_prediction_rows(rows: list[dict], threshold: float = 0.5) -> dict:
    y_true = [labels_to_multihot(row["gold_error_labels"]) for row in rows]
    y_pred = [[row["scores"][label] for label in ERROR_LABELS] for row in rows]
    return compute_multilabel_metrics(y_true, y_pred, threshold=threshold)


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise SystemExit(
            f"No trained model found at {model_dir}. Run scripts/train_distilbert.py with --dry-run false first."
        )
    expected_any = ["config.json", "model.safetensors", "pytorch_model.bin"]
    if not any((model_dir / filename).exists() for filename in expected_any):
        raise SystemExit(
            f"{model_dir} does not look like a saved Hugging Face model directory. "
            "Expected config.json plus model weights."
        )


def _has_gold(example: dict) -> bool:
    return ("target" in example and isinstance(example["target"], dict) and "error_labels" in example["target"]) or (
        "error_labels" in example
    )


def _all_have_gold(examples: list[dict]) -> bool:
    return all(_has_gold(example) for example in examples)


if __name__ == "__main__":
    main()
