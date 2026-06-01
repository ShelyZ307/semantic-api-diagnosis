#!/usr/bin/env python3
"""Evaluate visible-rule and original RoBERTa baselines on the fixed Stage 8 sample."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.baselines.visible_rule_based import VisibleRuleBasedBaseline
from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


GROUPS = {
    "seen_sample": {
        "sample": "data/generated/stage8_llm_sample/seen_sample.jsonl",
        "roberta_predictions": "outputs/roberta_v1_seed42/evaluation/seen_test.predictions.jsonl",
    },
    "unseen_sample": {
        "sample": "data/generated/stage8_llm_sample/unseen_sample.jsonl",
        "roberta_predictions": "outputs/roberta_v1_seed42/evaluation/unseen_family_test.predictions.jsonl",
    },
    "contract_dependent_unseen_sample": {
        "sample": "data/generated/stage8_llm_sample/contract_dependent_unseen_sample.jsonl",
        "roberta_predictions": "outputs/roberta_v1_seed42/evaluation/contract_dependent_unseen_test.predictions.jsonl",
    },
}


def main() -> None:
    parser = ArgumentParser(description="Evaluate non-LLM baselines on the fixed Stage 8 sample.")
    parser.add_argument("--train", default="data/generated/final_train.jsonl")
    parser.add_argument("--output", default="data/generated/stage8_llm_sample/non_llm_sample_metrics.json")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    rule_baseline = VisibleRuleBasedBaseline().fit(train_rows)
    results = {"groups": {}, "combined": {}}
    combined_rows = []
    combined_rule_predictions = []
    combined_roberta_predictions = []
    for group, paths in GROUPS.items():
        rows = read_jsonl(paths["sample"])
        roberta_predictions = _roberta_predictions(rows, Path(paths["roberta_predictions"]))
        rule_predictions = rule_baseline.predict(rows)
        results["groups"][group] = {
            "visible_rule_baseline": evaluate_predictions(rows, rule_predictions),
            "original_roberta": evaluate_predictions(rows, roberta_predictions),
        }
        combined_rows.extend(rows)
        combined_rule_predictions.extend(rule_predictions)
        combined_roberta_predictions.extend(roberta_predictions)
    results["combined"] = {
        "visible_rule_baseline": evaluate_predictions(combined_rows, combined_rule_predictions),
        "original_roberta": evaluate_predictions(combined_rows, combined_roberta_predictions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Stage 8 non-LLM sample metrics to {output}")


def _roberta_predictions(rows: list[dict], predictions_path: Path) -> list[dict]:
    by_id = {row["id"]: row["predicted_target"] for row in read_jsonl(predictions_path)}
    return [by_id[row["id"]] for row in rows]


if __name__ == "__main__":
    main()
