#!/usr/bin/env python3
"""Generate Stage 7 unseen-family error analysis for original and weighted RoBERTa."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.evaluation.contract_dependence import classify_example
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


SEMANTIC_LABELS = [
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
]


def main() -> None:
    parser = ArgumentParser(description="Analyze unseen-family semantic errors for Stage 7.")
    parser.add_argument("--examples", default="data/generated/final_unseen_family_test.jsonl")
    parser.add_argument("--original-dir", default="outputs/roberta_v1_seed42/evaluation")
    parser.add_argument("--weighted-dir", default="outputs/roberta_weighted_v1_seed42/evaluation")
    parser.add_argument("--tokenizer-dir", default="outputs/roberta_v1_seed42/model")
    parser.add_argument("--output", default="docs/results/stage_7_unseen_error_analysis.md")
    args = parser.parse_args()

    examples = {row["id"]: row for row in read_jsonl(args.examples)}
    token_counts = _token_counts(examples, Path(args.tokenizer_dir))
    models = {
        "original RoBERTa": _load_predictions(Path(args.original_dir), examples, token_counts),
        "weighted-loss RoBERTa": _load_predictions(Path(args.weighted_dir), examples, token_counts),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(models), encoding="utf-8")
    print(f"Wrote Stage 7 unseen error analysis to {output}")


def _load_predictions(path: Path, examples: dict[str, dict], token_counts: dict[str, int]) -> dict:
    thresholds = json.loads((path / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    predictions = read_jsonl(path / "unseen_family_test.predictions.jsonl")
    rows = [{**row, "example": examples[row["id"]], "token_count": token_counts[row["id"]]} for row in predictions]
    return {"rows": rows, "thresholds": thresholds}


def _render(models: dict[str, dict]) -> str:
    lines = [
        "# Stage 7 Unseen-Family Error Analysis",
        "",
        "This report compares the original and positive-class-weighted RoBERTa checkpoints on `final_unseen_family_test.jsonl`. "
        "Thresholds were tuned on validation only.",
        "",
    ]
    for model_name, model in models.items():
        lines.extend(_model_section(model_name, model))
    lines.extend(["## Comparative Interpretation", ""])
    original = _stats(models["original RoBERTa"])
    weighted = _stats(models["weighted-loss RoBERTa"])
    lines.extend(
        [
            f"- Critical semantic false negatives: original `{original['semantic_fn']}`, weighted `{weighted['semantic_fn']}`.",
            f"- Threshold-near semantic misses: original `{original['near_miss']}`, weighted `{weighted['near_miss']}`. "
            "A threshold-near miss has a score at least half of its selected validation threshold.",
            f"- Semantic false positives: original `{original['semantic_fp']}`, weighted `{weighted['semantic_fp']}`.",
            "- Remaining misses with scores far below threshold are more consistent with unseen-template transfer failure than threshold calibration alone.",
            "- The endpoint-local policies use unseen field names, enum/status vocabularies, and domain wording. Those shifts remain a central generalization challenge.",
            "- Truncation is not a plausible explanation: no unseen example exceeds `max_length=512`.",
            "- Original RoBERTa shows semantic-label confusion: cross-field violations are sometimes predicted as state violations, while valid or non-semantic rows can receive cross-field or state predictions.",
            "- Weighted loss is rejected as an improvement. It removes semantic false negatives by predicting violations broadly, increasing semantic false positives from "
            f"`{original['semantic_fp']}` to `{weighted['semantic_fp']}`.",
            "- Provider-backed LLM results are unavailable because no API credential was present, so LLM baselines do not change the current conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def _model_section(model_name: str, model: dict) -> list[str]:
    stats = _stats(model)
    rows = [
        f"## {model_name}",
        "",
        f"- Semantic false negatives: `{stats['semantic_fn']}`.",
        f"- Semantic false positives: `{stats['semantic_fp']}`.",
        f"- Threshold-near semantic misses: `{stats['near_miss']}`.",
        f"- Far-below-threshold semantic misses: `{stats['semantic_fn'] - stats['near_miss']}`.",
        f"- Examples exceeding `max_length=512`: `{stats['truncated_examples']}`; semantic misses on those examples: `{stats['truncated_fn']}`.",
        f"- Endpoint-local policy fields observed in semantic misses: `{', '.join(stats['policy_fields']) or 'none'}`.",
        "",
        "### Errors By Semantic Label",
        "",
        "| label | false negatives | false positives | threshold-near misses |",
        "|---|---:|---:|---:|",
    ]
    for label in SEMANTIC_LABELS:
        rows.append(
            f"| {label} | {stats['fn_by_label'][label]} | {stats['fp_by_label'][label]} | "
            f"{stats['near_by_label'][label]} |"
        )
    rows.extend(
        [
            "",
            "### Errors By Endpoint Family",
            "",
            "| endpoint family | semantic false negatives | semantic false positives |",
            "|---|---:|---:|",
        ]
    )
    families = sorted(set(stats["fn_by_family"]) | set(stats["fp_by_family"]))
    for family in families:
        rows.append(f"| {family} | {stats['fn_by_family'][family]} | {stats['fp_by_family'][family]} |")
    rows.extend(
        [
            "",
            "### Error Context",
            "",
            "| context | semantic false negatives |",
            "|---|---:|",
        ]
    )
    for context, count in sorted(stats["fn_context"].items()):
        rows.append(f"| {context} | {count} |")
    rows.extend(["", "### Sample Semantic False Negatives", ""])
    rows.extend(_sample_rows(stats["false_negatives"]))
    rows.extend(["", "### Sample Semantic False Positives", ""])
    rows.extend(_sample_rows(stats["false_positives"]))
    rows.append("")
    return rows


def _stats(model: dict) -> dict:
    fn_by_label = Counter()
    fp_by_label = Counter()
    near_by_label = Counter()
    fn_by_family = Counter()
    fp_by_family = Counter()
    fn_context = Counter()
    false_negatives = []
    false_positives = []
    truncated_examples = sum(row["token_count"] > 512 for row in model["rows"])
    truncated_fn = 0
    policy_fields = set()
    thresholds = model["thresholds"]
    for row in model["rows"]:
        example = row["example"]
        gold = set(row["gold_target"]["error_labels"])
        predicted = set(row["predicted_target"]["error_labels"])
        missed = sorted((gold - predicted) & set(SEMANTIC_LABELS))
        added = sorted((predicted - gold) & set(SEMANTIC_LABELS))
        for label in missed:
            score = row["scores"][label]
            fn_by_label[label] += 1
            fn_by_family[row["endpoint_family"]] += 1
            if score >= thresholds[label] / 2:
                near_by_label[label] += 1
            fn_context[f"complexity={example['generation_metadata']['error_complexity']}"] += 1
            fn_context[f"severity={example['target']['severity_bucket']}"] += 1
            fn_context[f"dependence={classify_example(example)}"] += 1
            fn_context[f"constraint_type={_constraint_type(label)}"] += 1
            if row["token_count"] > 512:
                truncated_fn += 1
            for policy in example["endpoint_contract"].get("policy", {}).values():
                if isinstance(policy, dict) and policy.get("field"):
                    policy_fields.add(policy["field"])
            false_negatives.append(_sample(row, label, score, thresholds[label]))
        for label in added:
            fp_by_label[label] += 1
            fp_by_family[row["endpoint_family"]] += 1
            false_positives.append(_sample(row, label, row["scores"][label], thresholds[label]))
    return {
        "semantic_fn": sum(fn_by_label.values()),
        "semantic_fp": sum(fp_by_label.values()),
        "near_miss": sum(near_by_label.values()),
        "fn_by_label": fn_by_label,
        "fp_by_label": fp_by_label,
        "near_by_label": near_by_label,
        "fn_by_family": fn_by_family,
        "fp_by_family": fp_by_family,
        "fn_context": fn_context,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "truncated_examples": truncated_examples,
        "truncated_fn": truncated_fn,
        "policy_fields": sorted(policy_fields),
    }


def _sample(row: dict, label: str, score: float, threshold: float) -> dict:
    return {
        "id": row["id"],
        "family": row["endpoint_family"],
        "label": label,
        "score": score,
        "threshold": threshold,
        "gold": row["gold_target"]["error_labels"],
        "predicted": row["predicted_target"]["error_labels"],
    }


def _sample_rows(samples: list[dict], limit: int = 8) -> list[str]:
    if not samples:
        return ["- None."]
    return [
        f"- `{row['id']}` (`{row['family']}`): `{row['label']}` score `{row['score']:.3f}` "
        f"vs threshold `{row['threshold']:.3f}`; gold `{row['gold']}`; predicted `{row['predicted']}`."
        for row in samples[:limit]
    ]


def _constraint_type(label: str) -> str:
    return {
        "semantic_cross_field_violation": "semantic_cross_field",
        "semantic_domain_constraint_violation": "semantic_domain_or_contract_local",
        "semantic_state_violation": "semantic_state_or_contract_local",
    }[label]


def _token_counts(examples: dict[str, dict], tokenizer_dir: Path) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return {
        example_id: len(tokenizer(example["serialized_input"], truncation=False)["input_ids"])
        for example_id, example in examples.items()
    }


if __name__ == "__main__":
    main()
