#!/usr/bin/env python3
"""Evaluate a fine-tuned encoder model on generated JSONL examples."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.training.data import get_error_labels, get_serialized_input, load_jsonl
from semantic_api_diagnosis.training.evaluation import (
    add_thresholded_prediction,
    compute_metrics_for_prediction_rows,
)
from semantic_api_diagnosis.training.labels import ERROR_LABELS
from semantic_api_diagnosis.serialization.jsonl import write_jsonl


def main() -> None:
    parser = ArgumentParser(description="Evaluate a fine-tuned model on JSONL examples.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds-json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    thresholds = _load_thresholds(args.thresholds_json) if args.thresholds_json else args.threshold

    model_dir = Path(args.model_dir)
    _validate_model_dir(model_dir)
    examples = load_jsonl(args.input_jsonl)
    predictions = predict_examples(
        model_dir=model_dir,
        examples=examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    predictions = [add_thresholded_prediction(row, thresholds) for row in predictions]
    output_path = Path(args.output_jsonl)
    write_jsonl(predictions, output_path)
    if _all_have_gold(examples):
        metrics = compute_metrics_for_prediction_rows(predictions, thresholds=thresholds)
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote predictions to {output_path}")


def predict_examples(
    model_dir: Path,
    examples: list[dict],
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
    device = _device(torch)
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            texts = [get_serialized_input(example) for example in batch]
            encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(**encoded)
            probabilities = torch.sigmoid(outputs.logits).cpu().tolist()
            for example, scores in zip(batch, probabilities):
                rows.append(format_prediction_row(example, scores))
    return rows


def format_prediction_row(example: dict, scores: list[float], threshold: float | None = None) -> dict:
    if len(scores) != len(ERROR_LABELS):
        raise ValueError(f"Expected {len(ERROR_LABELS)} scores, got {len(scores)}")
    gold_target = _gold_target(example) if _has_gold(example) else None
    row = {
        "id": example.get("id"),
        "source_split": example.get("split"),
        "endpoint_family": example.get("endpoint_family"),
        "gold_target": gold_target,
        "gold_error_labels": gold_target["error_labels"] if gold_target else [],
        "scores": {label: float(score) for label, score in zip(ERROR_LABELS, scores)},
    }
    return add_thresholded_prediction(row, threshold) if threshold is not None else row


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


def _gold_target(example: dict) -> dict:
    target = example.get("target", {})
    labels = get_error_labels(example)
    return {
        "error_labels": labels,
        "validity": target.get("validity", "invalid" if labels else "valid"),
        "severity_bucket": target.get("severity_bucket", "high" if labels else "none"),
    }


def _load_thresholds(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", payload)
    missing = [label for label in ERROR_LABELS if label not in thresholds]
    if missing:
        raise ValueError(f"Threshold file is missing labels: {', '.join(missing)}")
    return {label: float(thresholds[label]) for label in ERROR_LABELS}


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
