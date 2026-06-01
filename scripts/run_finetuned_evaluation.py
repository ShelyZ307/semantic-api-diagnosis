#!/usr/bin/env python3
"""Tune thresholds on validation and evaluate one encoder on all Version 1 views."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from evaluate_finetuned_model import predict_examples
from semantic_api_diagnosis.serialization.jsonl import write_jsonl
from semantic_api_diagnosis.training.evaluation import (
    add_thresholded_prediction,
    compute_metrics_for_prediction_rows,
    tune_thresholds,
)
from semantic_api_diagnosis.training.data import load_jsonl


EVALUATION_VIEWS = {
    "seen_test": "data/generated/final_seen_test.jsonl",
    "unseen_family_test": "data/generated/final_unseen_family_test.jsonl",
    "contract_dependent_seen_test": "data/generated/contract_dependent_seen_test.jsonl",
    "contract_dependent_unseen_test": "data/generated/contract_dependent_unseen_test.jsonl",
    "seen_test_request_only": "data/generated/shortcut_views_final_seen_test/request_only.jsonl",
    "seen_test_no_constraints": "data/generated/shortcut_views_final_seen_test/no_constraints.jsonl",
    "unseen_family_test_request_only": "data/generated/shortcut_views_final_unseen_family_test/request_only.jsonl",
    "unseen_family_test_no_constraints": "data/generated/shortcut_views_final_unseen_family_test/no_constraints.jsonl",
}


def main() -> None:
    parser = ArgumentParser(description="Evaluate a saved fine-tuned model on Version 1 views.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--validation", default="data/generated/final_validation.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold-mode", choices=["global", "per_label"], default="per_label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = _predict(args, Path(args.validation))
    threshold_payload = tune_thresholds(validation_rows, mode=args.threshold_mode)
    (output_dir / "thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_evaluation("validation", validation_rows, threshold_payload["thresholds"], output_dir)

    summary = {}
    all_rows = {}
    for name, path in EVALUATION_VIEWS.items():
        rows = _predict(args, Path(path))
        all_rows[name] = rows
        summary[name] = _write_evaluation(name, rows, threshold_payload["thresholds"], output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "error_analysis_sample.md").write_text(
        _error_analysis(all_rows, threshold_payload["thresholds"]),
        encoding="utf-8",
    )
    print(f"Wrote threshold and evaluation artifacts to {output_dir}")


def _predict(args, path: Path) -> list[dict]:
    return predict_examples(
        model_dir=Path(args.model_dir),
        examples=load_jsonl(path),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def _write_evaluation(name: str, rows: list[dict], thresholds: dict[str, float], output_dir: Path) -> dict:
    thresholded = [add_thresholded_prediction(row, thresholds) for row in rows]
    write_jsonl(thresholded, output_dir / f"{name}.predictions.jsonl")
    metrics = {
        "default_0_5": compute_metrics_for_prediction_rows(rows, 0.5),
        "tuned": compute_metrics_for_prediction_rows(rows, thresholds),
    }
    (output_dir / f"{name}.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Evaluated {name}: {len(rows)} examples")
    return metrics


def _error_analysis(all_rows: dict[str, list[dict]], thresholds: dict[str, float], limit: int = 12) -> str:
    lines = ["# Fine-Tuned Semantic False-Negative Sample", ""]
    for split in ["seen_test", "unseen_family_test"]:
        lines.extend([f"## {split}", ""])
        selected = 0
        for row in all_rows[split]:
            gold = set(row["gold_target"]["error_labels"])
            predicted = set(add_thresholded_prediction(row, thresholds)["predicted_target"]["error_labels"])
            missed = sorted(label for label in gold - predicted if label.startswith("semantic_"))
            if not missed:
                continue
            lines.extend(
                [
                    f"- id: {row['id']}",
                    f"  endpoint_family: {row['endpoint_family']}",
                    f"  missed_semantic_labels: {missed}",
                    f"  predicted_error_labels: {sorted(predicted)}",
                    "  scores: "
                    + ", ".join(
                        f"{label}={score:.3f}"
                        for label, score in row["scores"].items()
                        if label.startswith("semantic_")
                    ),
                ]
            )
            selected += 1
            if selected >= limit:
                break
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
