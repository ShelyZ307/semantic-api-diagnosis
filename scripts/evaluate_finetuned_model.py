#!/usr/bin/env python3
"""Scaffold for evaluating a fine-tuned encoder model."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.training.data import build_training_records


def main() -> None:
    parser = ArgumentParser(description="Evaluate a fine-tuned model on JSONL examples.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    records = build_training_records(args.input_jsonl)
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"No trained model found at {model_dir}. Run scripts/train_distilbert.py with --dry-run false first."
        )
    raise NotImplementedError(
        "Fine-tuned model prediction is scaffolded but not implemented yet. "
        f"Loaded {len(records)} records and found model directory {model_dir}."
    )


if __name__ == "__main__":
    main()
