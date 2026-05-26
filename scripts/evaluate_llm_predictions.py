#!/usr/bin/env python3
"""Evaluate saved LLM baseline predictions."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


def main() -> None:
    parser = ArgumentParser(description="Evaluate LLM baseline prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    gold = [{"target": row["gold"]} for row in rows]
    predictions = [
        row["parsed_prediction"] if row.get("parsed_prediction") else _empty_prediction()
        for row in rows
    ]
    metrics = evaluate_predictions(gold, predictions)
    parse_errors = sum(1 for row in rows if row.get("parse_error"))
    results = {
        "num_examples": len(rows),
        "parse_errors": parse_errors,
        "parse_error_rate": parse_errors / len(rows) if rows else 0.0,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(results), encoding="utf-8")
    print(f"Wrote LLM evaluation results to {output}")
    print(f"Wrote markdown report to {output.with_suffix('.md')}")


def _empty_prediction() -> dict:
    return {"validity": "valid", "error_labels": [], "severity_bucket": "none"}


def _markdown(results: dict) -> str:
    metrics = results["metrics"]
    return "\n".join(
        [
            "# LLM Baseline Evaluation",
            "",
            f"examples: {results['num_examples']}",
            f"parse_error_rate: {results['parse_error_rate']:.3f}",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| all-label micro-F1 | {metrics['all_labels']['micro_f1']:.3f} |",
            f"| all-label macro-F1 | {metrics['all_labels']['macro_f1']:.3f} |",
            f"| semantic micro-F1 | {metrics['semantic_labels']['micro_f1']:.3f} |",
            f"| semantic macro-F1 | {metrics['semantic_labels']['macro_f1']:.3f} |",
            f"| exact match | {metrics['exact_match_label_set_accuracy']:.3f} |",
            f"| validity accuracy | {metrics['validity']['accuracy']:.3f} |",
            f"| severity accuracy | {metrics['severity_bucket']['accuracy']:.3f} |",
            f"| critical semantic miss rate | {_fmt(metrics['critical_semantic_error_miss_rate'])} |",
        ]
    ) + "\n"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
