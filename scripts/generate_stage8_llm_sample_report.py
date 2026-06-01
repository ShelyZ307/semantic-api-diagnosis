#!/usr/bin/env python3
"""Generate the Stage 8 sampled LLM baseline report."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path


GROUP_LABELS = {
    "seen_sample": "seen sample",
    "unseen_sample": "unseen sample",
    "contract_dependent_unseen_sample": "contract-dependent unseen sample",
    "combined": "combined sample",
}
GROUP_ORDER = ["seen_sample", "unseen_sample", "contract_dependent_unseen_sample"]
MODEL_LABELS = {
    "visible_rule_baseline": "visible rule baseline",
    "original_roberta": "original RoBERTa",
    "zero_shot_llm": "zero-shot LLM",
    "few_shot_llm": "few-shot LLM",
}
SEMANTIC_LABELS = [
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
]


def main() -> None:
    parser = ArgumentParser(description="Generate Stage 8 sampled LLM baseline report.")
    parser.add_argument("--sample-dir", default="data/generated/stage8_llm_sample")
    parser.add_argument("--output", default="docs/results/stage_8_llm_sample_baselines.md")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    distribution = _read_json(sample_dir / "sample_distribution.json")
    non_llm = _read_json(sample_dir / "non_llm_sample_metrics.json")
    llm_results = {
        "zero_shot_llm": _optional_json(sample_dir / "zero_shot_metrics.json"),
        "few_shot_llm": _optional_json(sample_dir / "few_shot_metrics.json"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(distribution, non_llm, llm_results), encoding="utf-8")
    print(f"Wrote Stage 8 LLM sample report to {output}")


def _render(distribution: dict, non_llm: dict, llm_results: dict) -> str:
    lines = [
        "# Stage 8: Sampled Real LLM Baselines",
        "",
        "## Sample Description",
        "",
        f"- Total sample size: `{distribution['combined']['n']}`.",
        "- Sampling method: greedy stratified coverage of validity, multi-label cases, semantic labels, and structural labels, followed by seeded random fill.",
        "- Seed: `808`.",
        "- Source groups: `40` seen, `40` unseen-family, `40` contract-dependent unseen.",
        "- Limitation: this is a cost-controlled sample, so it complements but does not replace the full-test encoder and rule-baseline evaluation.",
        "",
    ]
    for group in GROUP_ORDER:
        stats = distribution["groups"][group]
        lines.extend(
            [
                f"### {GROUP_LABELS[group]}",
                "",
                f"- Validity: `{stats['validity']}`.",
                f"- Labels: `{stats['labels']}`.",
                f"- Semantic labels: `{stats['semantic_labels']}`.",
                f"- Endpoint families: `{stats['endpoint_families']}`.",
                f"- Single vs multi-error: `{stats['error_complexity']}`.",
                *(
                    [
                        "- Note: this source subset contains no `semantic_cross_field_violation` examples, so that semantic label cannot be represented here."
                    ]
                    if group == "contract_dependent_unseen_sample"
                    else []
                ),
                "",
            ]
        )
    lines.extend(_main_table(non_llm, llm_results))
    lines.extend(_semantic_table(non_llm, llm_results))
    lines.extend(_llm_quality(llm_results))
    lines.extend(_interpretation(non_llm, llm_results))
    return "\n".join(lines)


def _main_table(non_llm: dict, llm_results: dict) -> list[str]:
    lines = [
        "## Main Sampled Comparison",
        "",
        "| group | model | n | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in [*GROUP_ORDER, "combined"]:
        for model, metrics in _group_metrics(non_llm, llm_results, group):
            lines.append(_main_row(group, model, metrics))
    lines.append("")
    return lines


def _semantic_table(non_llm: dict, llm_results: dict) -> list[str]:
    lines = [
        "## Per-Semantic-Label F1",
        "",
        "| group | model | cross-field F1 | domain F1 | state F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for group in [*GROUP_ORDER, "combined"]:
        for model, metrics in _group_metrics(non_llm, llm_results, group):
            lines.append(_semantic_row(group, model, metrics))
    lines.append("")
    return lines


def _llm_quality(llm_results: dict) -> list[str]:
    lines = [
        "## LLM Parse And Usage Quality",
        "",
        "| model | group | parse failure rate | invalid JSON rate | transport error rate | usage totals |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, results in llm_results.items():
        if results is None:
            lines.append(f"| {MODEL_LABELS[model]} | all | n/a | n/a | n/a | no provider-backed artifact |")
            continue
        for group in [*GROUP_ORDER, "combined"]:
            item = _result_for_group(results, group)
            lines.append(
                f"| {MODEL_LABELS[model]} | {GROUP_LABELS[group]} | "
                f"{item['parse_failure_rate']:.3f} | {item['invalid_json_rate']:.3f} | "
                f"{item['transport_error_rate']:.3f} | `{item['usage']['totals']}` |"
            )
    lines.append("")
    return lines


def _interpretation(non_llm: dict, llm_results: dict) -> list[str]:
    lines = ["## Interpretation", ""]
    roberta_unseen = _metric(_non_llm_group(non_llm, "unseen_sample")["original_roberta"], "semantic_macro")
    rule_unseen = _metric(_non_llm_group(non_llm, "unseen_sample")["visible_rule_baseline"], "semantic_macro")
    lines.append(
        f"- On this fixed sample, original RoBERTa unseen semantic macro-F1 is `{roberta_unseen:.3f}` and the visible rule baseline is `{rule_unseen:.3f}`."
    )
    if llm_results["zero_shot_llm"] is None or llm_results["few_shot_llm"] is None:
        lines.append(
            "- Real zero-shot/few-shot conclusions are still blocked because provider-backed LLM artifacts are missing. "
            "Do not treat mock or missing rows as scientific evidence."
        )
    else:
        zero = _metric(_result_for_group(llm_results["zero_shot_llm"], "unseen_sample")["metrics"], "semantic_macro")
        few = _metric(_result_for_group(llm_results["few_shot_llm"], "unseen_sample")["metrics"], "semantic_macro")
        lines.append(f"- Zero-shot LLM unseen semantic macro-F1 is `{zero:.3f}`; few-shot is `{few:.3f}`.")
        lines.append(_comparison_sentence("zero-shot LLM", zero, "original RoBERTa", roberta_unseen))
        lines.append(_comparison_sentence("few-shot LLM", few, "original RoBERTa", roberta_unseen))
        lines.append(_comparison_sentence("few-shot LLM", few, "zero-shot LLM", zero))
        lines.append(_comparison_sentence("best sampled LLM", max(zero, few), "visible rule baseline", rule_unseen))
    lines.extend(
        [
            "- Recommended framing: RoBERTa learns contract-sensitive behavior and performs strongly in-domain, but unseen-family transfer remains weak. "
            "The project contribution is a controlled benchmark and analysis of contract-dependent semantic API diagnosis, not a claim that encoders solve unseen semantic generalization.",
            "",
        ]
    )
    return lines


def _group_metrics(non_llm: dict, llm_results: dict, group: str) -> list[tuple[str, dict | None]]:
    rows: list[tuple[str, dict | None]] = []
    non_llm_group = _non_llm_group(non_llm, group)
    rows.extend(
        [
            ("visible_rule_baseline", non_llm_group["visible_rule_baseline"]),
            ("original_roberta", non_llm_group["original_roberta"]),
        ]
    )
    for model, results in llm_results.items():
        rows.append((model, None if results is None else _result_for_group(results, group)["metrics"]))
    return rows


def _non_llm_group(non_llm: dict, group: str) -> dict:
    return non_llm["combined"] if group == "combined" else non_llm["groups"][group]


def _result_for_group(results: dict, group: str) -> dict:
    return results["combined"] if group == "combined" else results["groups"][group]


def _main_row(group: str, model: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"| {GROUP_LABELS[group]} | {MODEL_LABELS[model]} | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {GROUP_LABELS[group]} | {MODEL_LABELS[model]} | {_group_n(group)} | "
        f"{_metric(metrics, 'micro'):.3f} | {_metric(metrics, 'semantic_macro'):.3f} | "
        f"{metrics['exact_match_label_set_accuracy']:.3f} | {_fmt(metrics['critical_semantic_error_miss_rate'])} |"
    )


def _semantic_row(group: str, model: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"| {GROUP_LABELS[group]} | {MODEL_LABELS[model]} | n/a | n/a | n/a |"
    per_label = metrics["semantic_labels"]["per_label"]
    return (
        f"| {GROUP_LABELS[group]} | {MODEL_LABELS[model]} | "
        f"{per_label['semantic_cross_field_violation']['f1']:.3f} | "
        f"{per_label['semantic_domain_constraint_violation']['f1']:.3f} | "
        f"{per_label['semantic_state_violation']['f1']:.3f} |"
    )


def _metric(metrics: dict, name: str) -> float:
    if name == "micro":
        return metrics["all_labels"]["micro_f1"]
    if name == "semantic_macro":
        return metrics["semantic_labels"]["macro_f1"]
    raise KeyError(name)


def _group_n(group: str) -> int:
    return 120 if group == "combined" else 40


def _comparison_sentence(left_name: str, left: float, right_name: str, right: float) -> str:
    verb = "outperforms" if left > right else "does not outperform"
    return f"- {left_name} {verb} {right_name} on sampled unseen semantic macro-F1 (`{left:.3f}` vs `{right:.3f}`)."


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict | None:
    return _read_json(path) if path.exists() else None


if __name__ == "__main__":
    main()
