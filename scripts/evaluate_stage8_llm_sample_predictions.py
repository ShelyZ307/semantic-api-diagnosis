#!/usr/bin/env python3
"""Evaluate Stage 8 LLM predictions by sample group and combined sample."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


GROUPS = ["seen_sample", "unseen_sample", "contract_dependent_unseen_sample"]


def main() -> None:
    parser = ArgumentParser(description="Evaluate fixed-sample Stage 8 LLM predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    results = {"groups": {}, "combined": _evaluate(rows)}
    for group in GROUPS:
        group_rows = [row for row in rows if row.get("source_split") == group]
        results["groups"][group] = _evaluate(group_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Stage 8 LLM sample metrics to {output}")


def _evaluate(rows: list[dict]) -> dict:
    metrics = evaluate_predictions(_gold_examples(rows), [_prediction(row) for row in rows]) if rows else {}
    parse_errors = Counter(_failure_type(row) for row in rows)
    total = len(rows)
    usage = _usage(rows)
    return {
        "n": total,
        "metrics": metrics,
        "parse_failure_count": parse_errors["parse_error"],
        "parse_failure_rate": _rate(parse_errors["parse_error"], total),
        "invalid_json_count": parse_errors["invalid_json"],
        "invalid_json_rate": _rate(parse_errors["invalid_json"], total),
        "transport_error_count": parse_errors["transport_error"],
        "transport_error_rate": _rate(parse_errors["transport_error"], total),
        "usage": usage,
    }


def _gold_examples(rows: list[dict]) -> list[dict]:
    return [{"target": row["gold"]} for row in rows]


def _prediction(row: dict) -> dict:
    return row.get("parsed_prediction") or {
        "validity": "valid",
        "error_labels": [],
        "severity_bucket": "none",
    }


def _failure_type(row: dict) -> str:
    if row.get("transport_error"):
        return "transport_error"
    parse_error = row.get("parse_error")
    if not parse_error:
        return "ok"
    if "json" in parse_error.lower() or "expecting" in parse_error.lower():
        return "invalid_json"
    return "parse_error"


def _usage(rows: list[dict]) -> dict:
    totals = Counter()
    responses_with_usage = 0
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        responses_with_usage += 1
        for key, value in usage.items():
            if isinstance(value, int | float):
                totals[key] += value
    return {"responses_with_usage": responses_with_usage, "totals": dict(sorted(totals.items()))}


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


if __name__ == "__main__":
    main()
