#!/usr/bin/env python3
"""Evaluate transparent baselines on final Version 1 dataset splits."""

from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.baselines.family_frequency import FamilyFrequencyBaseline
from semantic_api_diagnosis.baselines.majority import MajorityBaseline
from semantic_api_diagnosis.baselines.visible_rule_based import VisibleRuleBasedBaseline
from semantic_api_diagnosis.evaluation.contract_dependence import classify_example
from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.quality_gate import create_shortcut_views
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


def main() -> None:
    parser = ArgumentParser(description="Evaluate baseline models for semantic API diagnosis.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--seen-test", required=True)
    parser.add_argument("--unseen-test", required=True)
    parser.add_argument("--output", default="data/generated/baseline_results.json")
    parser.add_argument("--frequent-label-threshold", type=float, default=0.3)
    parser.add_argument("--seen-test-request-only")
    parser.add_argument("--seen-test-contract-only")
    parser.add_argument("--seen-test-no-constraints")
    parser.add_argument("--seen-test-family-only")
    parser.add_argument("--unseen-test-request-only")
    parser.add_argument("--unseen-test-contract-only")
    parser.add_argument("--unseen-test-no-constraints")
    parser.add_argument("--unseen-test-family-only")
    parser.add_argument("--contract-dependent-seen-test")
    parser.add_argument("--contract-dependent-unseen-test")
    args = parser.parse_args()

    train = read_jsonl(args.train)
    splits = {
        "validation": read_jsonl(args.validation),
        "seen_test": read_jsonl(args.seen_test),
        "unseen_family_test": read_jsonl(args.unseen_test),
    }
    baselines = {
        "majority_empty": MajorityBaseline(label_threshold=None).fit(train),
        "majority_frequent": MajorityBaseline(label_threshold=args.frequent_label_threshold).fit(train),
        "family_frequency": FamilyFrequencyBaseline(label_threshold=args.frequent_label_threshold).fit(train),
        "visible_rule_based": VisibleRuleBasedBaseline().fit(train),
    }
    results = {"splits": {}, "hard_subsets": {}, "shortcut_views": {}, "contract_dependence": {}, "messages": []}
    for split_name, rows in splits.items():
        results["splits"][split_name] = _evaluate_baselines(baselines, rows)

    hard_subset_paths = _resolve_hard_subset_paths(args)
    for subset_name, path in hard_subset_paths.items():
        if path.exists():
            results["hard_subsets"][subset_name] = _evaluate_baselines(baselines, read_jsonl(path))
        else:
            message = (
                f"Hard subset file missing for {subset_name}: {path}. "
                "Run scripts/analyze_contract_dependence.py first to create contract-dependent subset files."
            )
            results["messages"].append(message)
            print(message)

    shortcut_paths = _resolve_shortcut_paths(args)
    for view_name, path in shortcut_paths.items():
        rows = read_jsonl(path)
        results["shortcut_views"][view_name] = _evaluate_baselines(
            {
                "family_frequency": baselines["family_frequency"],
                "visible_rule_based": baselines["visible_rule_based"],
            },
            rows,
        )

    results["contract_dependence"] = _contract_dependence_summary(splits, results["hard_subsets"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(_markdown_report(results), encoding="utf-8")
    print(f"Wrote baseline results to {output_path}")
    print(f"Wrote markdown report to {markdown_path}")


def _evaluate_baselines(baselines: dict, rows: list[dict]) -> dict:
    output = {}
    for name, baseline in baselines.items():
        output[name] = evaluate_predictions(rows, baseline.predict(rows))
    return output


def _resolve_shortcut_paths(args) -> dict[str, Path]:
    seen_paths = _ensure_shortcuts(
        args.seen_test,
        Path(args.seen_test).parent / "shortcut_views_final_seen_test",
        {
            "request_only": args.seen_test_request_only,
            "contract_only": args.seen_test_contract_only,
            "no_constraints": args.seen_test_no_constraints,
            "family_only": args.seen_test_family_only,
        },
    )
    unseen_paths = _ensure_shortcuts(
        args.unseen_test,
        Path(args.unseen_test).parent / "shortcut_views_final_unseen_family_test",
        {
            "request_only": args.unseen_test_request_only,
            "contract_only": args.unseen_test_contract_only,
            "no_constraints": args.unseen_test_no_constraints,
            "family_only": args.unseen_test_family_only,
        },
    )
    paths = {}
    for view_name, path in seen_paths.items():
        paths[f"seen_test_{view_name}"] = path
    for view_name, path in unseen_paths.items():
        paths[f"unseen_family_test_{view_name}"] = path
    return paths


def _resolve_hard_subset_paths(args) -> dict[str, Path]:
    base_dir = Path(args.output).parent
    return {
        "contract_dependent_seen_test": Path(args.contract_dependent_seen_test)
        if args.contract_dependent_seen_test
        else base_dir / "contract_dependent_seen_test.jsonl",
        "contract_dependent_unseen_test": Path(args.contract_dependent_unseen_test)
        if args.contract_dependent_unseen_test
        else base_dir / "contract_dependent_unseen_test.jsonl",
    }


def _ensure_shortcuts(dataset_path: str, output_dir: Path, provided: dict[str, str | None]) -> dict[str, Path]:
    expected = {
        view_name: Path(path) if path else output_dir / f"{view_name}.jsonl"
        for view_name, path in provided.items()
    }
    if not all(path.exists() for path in expected.values()):
        created = create_shortcut_views(dataset_path, output_dir)
        for view_name, path in created.items():
            expected[view_name] = path
    return expected


def _markdown_report(results: dict) -> str:
    lines = ["# Baseline Evaluation Results", ""]
    lines.extend(_interpretation_block())
    lines.extend(["", "## Full Split Results", ""])
    lines.extend(_main_table(results["splits"]))
    lines.extend(["", "## Semantic Label Performance", ""])
    lines.extend(_semantic_table(results["splits"]))
    lines.extend(["", "## Critical Semantic Miss Rate", ""])
    lines.extend(_miss_rate_table(results["splits"]))
    lines.extend(["", "## Hard Contract-Dependent Subset Results", ""])
    lines.extend(_hard_subset_section(results["hard_subsets"], results["messages"]))
    lines.extend(["", "## Shortcut Comparison", ""])
    lines.extend(_shortcut_comparison_table(results["splits"], results["shortcut_views"]))
    lines.extend(["", "## Contract-Dependence Summary", ""])
    lines.extend(_contract_dependence_section(results["contract_dependence"], results["hard_subsets"]))
    return "\n".join(lines) + "\n"


def _interpretation_block() -> list[str]:
    return [
        "## Interpretation",
        "",
        "- Majority and family-frequency baselines are weak, as expected for a multi-label diagnosis task.",
        "- The visible rule-based baseline is strong on full and request-only views, indicating many request-obvious examples.",
        "- Contract-dependent subsets provide a harder evaluation target focused on examples that need endpoint policy or constraints.",
        "- Future model results should be reported both on the full test sets and on the contract-dependent subsets.",
    ]


def _main_table(split_results: dict) -> list[str]:
    rows = [
        "| split | baseline | micro-F1 | macro-F1 | exact match | validity acc | severity acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, baselines in split_results.items():
        for baseline, metrics in baselines.items():
            rows.append(
                "| {split} | {baseline} | {micro:.3f} | {macro:.3f} | {exact:.3f} | {validity:.3f} | {severity:.3f} |".format(
                    split=split,
                    baseline=baseline,
                    micro=metrics["all_labels"]["micro_f1"],
                    macro=metrics["all_labels"]["macro_f1"],
                    exact=metrics["exact_match_label_set_accuracy"],
                    validity=metrics["validity"]["accuracy"],
                    severity=metrics["severity_bucket"]["accuracy"],
                )
            )
    return rows


def _semantic_table(split_results: dict) -> list[str]:
    rows = ["| split | baseline | semantic micro-F1 | semantic macro-F1 | per semantic label F1 |", "|---|---:|---:|---:|---|"]
    for split, baselines in split_results.items():
        for baseline, metrics in baselines.items():
            per_label = ", ".join(
                f"{label}={values['f1']:.3f}"
                for label, values in metrics["semantic_labels"]["per_label"].items()
            )
            rows.append(
                f"| {split} | {baseline} | {metrics['semantic_labels']['micro_f1']:.3f} | "
                f"{metrics['semantic_labels']['macro_f1']:.3f} | {per_label} |"
            )
    return rows


def _miss_rate_table(split_results: dict) -> list[str]:
    rows = ["| split | baseline | semantic miss rate | high-severity semantic miss rate |", "|---|---:|---:|---:|"]
    for split, baselines in split_results.items():
        for baseline, metrics in baselines.items():
            rows.append(
                f"| {split} | {baseline} | {_fmt(metrics['critical_semantic_error_miss_rate'])} | "
                f"{_fmt(metrics['critical_semantic_error_miss_rate_high_severity'])} |"
            )
    return rows


def _shortcut_table(shortcut_results: dict) -> list[str]:
    rows = ["| view | baseline | micro-F1 | semantic macro-F1 | exact match | validity acc |", "|---|---:|---:|---:|---:|---:|"]
    for view_name, baselines in shortcut_results.items():
        for baseline, metrics in baselines.items():
            rows.append(
                f"| {view_name} | {baseline} | {metrics['all_labels']['micro_f1']:.3f} | "
                f"{metrics['semantic_labels']['macro_f1']:.3f} | "
                f"{metrics['exact_match_label_set_accuracy']:.3f} | {metrics['validity']['accuracy']:.3f} |"
            )
    return rows


def _hard_subset_section(hard_subset_results: dict, messages: list[str]) -> list[str]:
    lines = []
    if hard_subset_results:
        lines.extend(_main_table(hard_subset_results))
        lines.extend(["", "### Hard Subset Semantic Performance", ""])
        lines.extend(_semantic_table(hard_subset_results))
        lines.extend(["", "### Hard Subset Critical Semantic Miss Rate", ""])
        lines.extend(_miss_rate_table(hard_subset_results))
    if messages:
        lines.extend(["", "### Missing Hard Subsets", ""])
        lines.extend([f"- {message}" for message in messages])
    if not hard_subset_results and not messages:
        lines.append("- No hard subset files were provided or discovered.")
    return lines


def _shortcut_comparison_table(split_results: dict, shortcut_results: dict) -> list[str]:
    rows = [
        "| evaluation | baseline | micro-F1 | semantic macro-F1 | exact match | validity acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split_name in ["seen_test", "unseen_family_test"]:
        if split_name in split_results and "visible_rule_based" in split_results[split_name]:
            rows.append(_view_row(f"{split_name}_full", "visible_rule_based", split_results[split_name]["visible_rule_based"]))
    for view_name, baselines in shortcut_results.items():
        if any(view_name.endswith(suffix) for suffix in ["request_only", "no_constraints", "family_only"]):
            for baseline, metrics in baselines.items():
                rows.append(_view_row(view_name, baseline, metrics))
    return rows


def _view_row(view_name: str, baseline: str, metrics: dict) -> str:
    return (
        f"| {view_name} | {baseline} | {metrics['all_labels']['micro_f1']:.3f} | "
        f"{metrics['semantic_labels']['macro_f1']:.3f} | "
        f"{metrics['exact_match_label_set_accuracy']:.3f} | {metrics['validity']['accuracy']:.3f} |"
    )


def _contract_dependence_summary(splits: dict, hard_subsets: dict) -> dict:
    summary = {}
    for split_name in ["seen_test", "unseen_family_test"]:
        rows = splits.get(split_name, [])
        bucket_counts = {}
        semantic_bucket_counts = {}
        semantic_total = 0
        for row in rows:
            bucket = classify_example(row)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if any(label.startswith("semantic_") for label in row["target"]["error_labels"]):
                semantic_total += 1
                semantic_bucket_counts[bucket] = semantic_bucket_counts.get(bucket, 0) + 1
        summary[split_name] = {
            "total": len(rows),
            "bucket_counts": bucket_counts,
            "semantic_total": semantic_total,
            "semantic_bucket_counts": semantic_bucket_counts,
        }
    for subset_name, baselines in hard_subsets.items():
        summary[subset_name] = {
            "visible_rule_based": baselines.get("visible_rule_based", {}),
        }
    return summary


def _contract_dependence_section(summary: dict, hard_subset_results: dict) -> list[str]:
    lines = [
        "### Bucket Distribution",
        "",
        "| split | valid | request_obvious | contract_dependent | mixed |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in ["seen_test", "unseen_family_test"]:
        data = summary.get(split_name, {})
        total = data.get("total", 0)
        counts = data.get("bucket_counts", {})
        lines.append(
            f"| {split_name} | {_count_pct(counts.get('valid', 0), total)} | "
            f"{_count_pct(counts.get('request_obvious', 0), total)} | "
            f"{_count_pct(counts.get('contract_dependent', 0), total)} | "
            f"{_count_pct(counts.get('mixed', 0), total)} |"
        )
    lines.extend(
        [
            "",
            "### Semantic Examples by Bucket",
            "",
            "| split | request_obvious | contract_dependent | mixed |",
            "|---|---:|---:|---:|",
        ]
    )
    for split_name in ["seen_test", "unseen_family_test"]:
        data = summary.get(split_name, {})
        total = data.get("semantic_total", 0)
        counts = data.get("semantic_bucket_counts", {})
        lines.append(
            f"| {split_name} | {_count_pct(counts.get('request_obvious', 0), total)} | "
            f"{_count_pct(counts.get('contract_dependent', 0), total)} | "
            f"{_count_pct(counts.get('mixed', 0), total)} |"
        )
    lines.extend(
        [
            "",
            "### Visible Rule-Based on Contract-Dependent Subsets",
            "",
            "| subset | semantic macro-F1 | semantic micro-F1 | critical semantic miss rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for subset_name, baselines in hard_subset_results.items():
        metrics = baselines.get("visible_rule_based")
        if not metrics:
            continue
        lines.append(
            f"| {subset_name} | {metrics['semantic_labels']['macro_f1']:.3f} | "
            f"{metrics['semantic_labels']['micro_f1']:.3f} | "
            f"{_fmt(metrics['critical_semantic_error_miss_rate'])} |"
        )
    return lines


def _count_pct(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else "0 (0.0%)"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
