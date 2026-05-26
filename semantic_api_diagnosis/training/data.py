"""JSONL loading and record preparation for fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_api_diagnosis.training.labels import labels_to_multihot


def load_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {jsonl_path}: {exc}") from exc
    return rows


def get_serialized_input(example: dict) -> str:
    if "serialized_input" not in example:
        raise ValueError("Example is missing required field: serialized_input")
    value = example["serialized_input"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Example serialized_input must be a non-empty string")
    return value


def get_error_labels(example: dict) -> list[str]:
    if "target" in example and isinstance(example["target"], dict) and "error_labels" in example["target"]:
        labels = example["target"]["error_labels"]
    elif "error_labels" in example:
        labels = example["error_labels"]
    else:
        raise ValueError("Example is missing error labels at target.error_labels or error_labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("error_labels must be a list of strings")
    return labels


def build_training_records(path: str | Path) -> list[dict]:
    records = []
    for index, example in enumerate(load_jsonl(path), start=1):
        try:
            records.append(
                {
                    "text": get_serialized_input(example),
                    "labels": labels_to_multihot(get_error_labels(example)),
                }
            )
        except ValueError as exc:
            example_id = example.get("id", f"line_{index}")
            raise ValueError(f"Invalid training example {example_id}: {exc}") from exc
    return records
