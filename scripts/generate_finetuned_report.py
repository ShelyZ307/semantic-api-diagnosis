#!/usr/bin/env python3
"""Generate the Version 1 fine-tuned encoder results report from saved artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


MAIN_VIEWS = [
    "seen_test",
    "unseen_family_test",
    "contract_dependent_seen_test",
    "contract_dependent_unseen_test",
]
SHORTCUT_VIEWS = [
    ("seen_test", "seen_test_request_only", "seen_test_no_constraints"),
    ("unseen_family_test", "unseen_family_test_request_only", "unseen_family_test_no_constraints"),
]
ERROR_LABELS = sorted(FIXED_LABEL_TAXONOMY)
SEMANTIC_LABELS = [label for label in ERROR_LABELS if label.startswith("semantic_")]


def main() -> None:
    parser = ArgumentParser(description="Generate the checked-in fine-tuned encoder report.")
    parser.add_argument("--baseline-results", default="docs/results/baseline_results.json")
    parser.add_argument("--distilbert-dir", default="outputs/distilbert_v1_seed42/evaluation")
    parser.add_argument("--roberta-dir", default="outputs/roberta_v1_seed42/evaluation")
    parser.add_argument("--output", default="docs/results/fine_tuned_model_results.md")
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_results))
    models = {
        "DistilBERT": _load_model_artifacts(Path(args.distilbert_dir)),
        "RoBERTa": _load_model_artifacts(Path(args.roberta_dir)),
    }
    available_models = {name: artifacts for name, artifacts in models.items() if artifacts is not None}
    if not available_models:
        parser.error("at least one completed fine-tuned evaluation directory is required")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_report(baseline, models), encoding="utf-8")
    print(f"Wrote fine-tuned model report to {output}")


def _render_report(baseline: dict, models: dict[str, dict | None]) -> str:
    split_sizes = {
        name: len(read_jsonl(Path(path)))
        for name, path in {
            "train": "data/generated/final_train.jsonl",
            "validation": "data/generated/final_validation.jsonl",
            "seen_test": "data/generated/final_seen_test.jsonl",
            "unseen_family_test": "data/generated/final_unseen_family_test.jsonl",
            "contract_dependent_seen_test": "data/generated/contract_dependent_seen_test.jsonl",
            "contract_dependent_unseen_test": "data/generated/contract_dependent_unseen_test.jsonl",
        }.items()
    }
    lines = [
        "# Fine-Tuned Encoder Results",
        "",
        "## Experiment Setup",
        "",
        "Version 1 uses the tightened contract-dependence dataset without adding endpoint families or changing the label taxonomy.",
        "",
        "| split | examples |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {size} |" for name, size in split_sizes.items())
    lines.extend(
        [
            "",
            f"- Labels: {len(ERROR_LABELS)} multi-label diagnosis targets: {', '.join(ERROR_LABELS)}.",
            "- Encoders: `distilbert-base-uncased` and `roberta-base` when a completed checkpoint is available.",
            "- Training: three epochs, fixed seed `42`, maximum sequence length `512`, validation on `final_validation.jsonl` only. DistilBERT used batch size `16`; RoBERTa used the CPU-feasible batch size `4`.",
            "- Thresholding: per-label thresholds selected on validation macro-F1 and reused unchanged for every test and shortcut view.",
            "- Auxiliary validity and severity metrics are derived from predicted error labels; there are no separate auxiliary heads.",
            "- LLM baseline code exists, but no real provider-backed zero-shot or few-shot result is available. Mock output is excluded from scientific comparisons.",
            "",
            "## Main Results",
            "",
            "| view | model | micro-F1 | macro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for view in MAIN_VIEWS:
        baseline_metrics = _baseline_metrics(baseline, view)
        lines.append(_main_row(view, "visible rule-based", baseline_metrics))
        for model_name, artifacts in models.items():
            if artifacts is None:
                lines.append(f"| {view} | {model_name} | n/a | n/a | n/a | n/a | n/a |")
            else:
                lines.append(_main_row(view, model_name, artifacts["summary"][view]["tuned"]))
        lines.append(f"| {view} | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |")
        lines.append(f"| {view} | few-shot LLM | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "Real zero-shot and few-shot LLM values are `n/a` because the current scaffold has no provider-backed experiment artifact.",
            "",
            "## Threshold Comparison",
            "",
            "| model | view | thresholding | micro-F1 | macro-F1 | semantic macro-F1 | exact match |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, artifacts in models.items():
        if artifacts is None:
            continue
        for view in ["seen_test", "unseen_family_test"]:
            for threshold_name in ["default_0_5", "tuned"]:
                metrics = artifacts["summary"][view][threshold_name]
                lines.append(
                    f"| {model_name} | {view} | {threshold_name} | {_metric(metrics, 'micro_f1')} | "
                    f"{_metric(metrics, 'macro_f1')} | {_semantic_metric(metrics)} | "
                    f"{metrics['exact_match_label_set_accuracy']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Shortcut Ablation",
            "",
            "| model | test split | full semantic macro-F1 | request-only | no constraints | request-only delta | no-constraints delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, artifacts in models.items():
        if artifacts is None:
            continue
        for full, request_only, no_constraints in SHORTCUT_VIEWS:
            full_score = artifacts["summary"][full]["tuned"]["semantic_labels"]["macro_f1"]
            request_score = artifacts["summary"][request_only]["tuned"]["semantic_labels"]["macro_f1"]
            no_constraints_score = artifacts["summary"][no_constraints]["tuned"]["semantic_labels"]["macro_f1"]
            lines.append(
                f"| {model_name} | {full} | {full_score:.3f} | {request_score:.3f} | "
                f"{no_constraints_score:.3f} | {request_score - full_score:+.3f} | "
                f"{no_constraints_score - full_score:+.3f} |"
            )
    lines.extend(["", "## Error Analysis", ""])
    for model_name, artifacts in models.items():
        lines.extend(_error_analysis(model_name, artifacts))
    lines.extend(
        [
            "## Auxiliary Metrics",
            "",
            "| model | view | validity accuracy | validity F1 | severity accuracy |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_name, artifacts in models.items():
        if artifacts is None:
            continue
        for view in ["seen_test", "unseen_family_test"]:
            metrics = artifacts["summary"][view]["tuned"]
            lines.append(
                f"| {model_name} | {view} | {metrics['validity']['accuracy']:.3f} | "
                f"{metrics['validity']['f1']:.3f} | {metrics['severity_bucket']['accuracy']:.3f} |"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(_interpretation(baseline, models))
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "Artifacts are saved under `outputs/distilbert_v1_seed42/` and `outputs/roberta_v1_seed42/` when each model run completes. "
            "Each evaluation directory contains `thresholds.json`, per-view metric JSON files, per-example predictions, `summary.json`, and an error-analysis sample.",
            "",
        ]
    )
    return "\n".join(lines)


def _error_analysis(model_name: str, artifacts: dict | None) -> list[str]:
    if artifacts is None:
        return [f"### {model_name}", "", "No completed checkpoint is available.", ""]
    metrics = artifacts["summary"]["unseen_family_test"]["tuned"]
    label_metrics = metrics["all_labels"]["per_label"]
    weakest = sorted(label_metrics.items(), key=lambda item: (item[1]["f1"], item[0]))[:3]
    rows = artifacts["predictions"]["unseen_family_test"]
    false_negatives = []
    cross_domain_confusions = 0
    for row in rows:
        gold = set(row["gold_target"]["error_labels"])
        predicted = set(row["predicted_target"]["error_labels"])
        missed = sorted((gold - predicted) & set(SEMANTIC_LABELS))
        if (
            "semantic_cross_field_violation" in gold
            and "semantic_domain_constraint_violation" in predicted - gold
        ) or (
            "semantic_domain_constraint_violation" in gold
            and "semantic_cross_field_violation" in predicted - gold
        ):
            cross_domain_confusions += 1
        if missed and len(false_negatives) < 3:
            false_negatives.append((row["id"], row["endpoint_family"], missed, sorted(predicted)))
    output = [
        f"### {model_name}",
        "",
        "Weakest unseen-family labels: "
        + ", ".join(f"`{label}` F1={values['f1']:.3f}" for label, values in weakest)
        + ".",
        "",
        f"Cross-field/domain substitution cases on unseen families: {cross_domain_confusions}.",
        "",
        "Sample unseen-family semantic false negatives:",
    ]
    if not false_negatives:
        output.append("- None.")
    else:
        for example_id, family, missed, predicted in false_negatives:
            output.append(
                f"- `{example_id}` (`{family}`): missed `{', '.join(missed)}`; predicted `{', '.join(predicted) or 'no errors'}`."
            )
    output.append("")
    return output


def _interpretation(baseline: dict, models: dict[str, dict | None]) -> list[str]:
    lines = []
    for model_name, artifacts in models.items():
        if artifacts is None:
            lines.append(f"- {model_name}: no completed experiment artifact; no claim can be made.")
            continue
        seen = artifacts["summary"]["seen_test"]["tuned"]
        unseen = artifacts["summary"]["unseen_family_test"]["tuned"]
        seen_baseline = _baseline_metrics(baseline, "seen_test")
        unseen_baseline = _baseline_metrics(baseline, "unseen_family_test")
        seen_delta = seen["semantic_labels"]["macro_f1"] - seen_baseline["semantic_labels"]["macro_f1"]
        unseen_delta = unseen["semantic_labels"]["macro_f1"] - unseen_baseline["semantic_labels"]["macro_f1"]
        request = artifacts["summary"]["seen_test_request_only"]["tuned"]["semantic_labels"]["macro_f1"]
        full = seen["semantic_labels"]["macro_f1"]
        lines.append(
            f"- {model_name}: semantic macro-F1 versus the visible rule baseline changes by "
            f"{seen_delta:+.3f} on seen families and {unseen_delta:+.3f} on unseen families. "
            f"Removing contract text changes seen semantic macro-F1 by {request - full:+.3f}."
        )
    lines.extend(
        [
            "- Do fine-tuned models beat the visible rule baseline? Not consistently. RoBERTa beats it on all-label seen-family metrics and on the contract-dependent seen subset, but not on semantic seen-family macro-F1 and not on unseen families.",
            "- Do fine-tuned models improve on contract-dependent examples? RoBERTa does on seen families. The unseen-family hard subset remains weak.",
            "- Do models rely on contract text? RoBERTa clearly does: removing constraints produces a large semantic macro-F1 drop on seen and unseen tests. DistilBERT shows only a small seen drop and an inconsistent unseen ablation.",
            "- Does Version 1 support the full research claim? No. The results support a narrower claim that a fine-tuned RoBERTa encoder learns contract-dependent diagnosis on seen families and partially transfers to unseen families, while the transparent rule baseline remains substantially stronger on unseen-family semantic diagnosis.",
        ]
    )
    return lines


def _main_row(view: str, model_name: str, metrics: dict) -> str:
    return (
        f"| {view} | {model_name} | {_metric(metrics, 'micro_f1')} | {_metric(metrics, 'macro_f1')} | "
        f"{_semantic_metric(metrics)} | {metrics['exact_match_label_set_accuracy']:.3f} | "
        f"{metrics['critical_semantic_error_miss_rate']:.3f} |"
    )


def _metric(metrics: dict, name: str) -> str:
    return f"{metrics.get(name, metrics['all_labels'][name]):.3f}"


def _semantic_metric(metrics: dict) -> str:
    return f"{metrics['semantic_labels']['macro_f1']:.3f}"


def _baseline_metrics(baseline: dict, view: str) -> dict:
    if view in baseline["splits"]:
        return baseline["splits"][view]["visible_rule_based"]
    return baseline["hard_subsets"][view]["visible_rule_based"]


def _load_model_artifacts(path: Path) -> dict | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    return {
        "summary": _read_json(summary_path),
        "thresholds": _read_json(path / "thresholds.json"),
        "predictions": {
            view: read_jsonl(path / f"{view}.predictions.jsonl")
            for view in MAIN_VIEWS
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
