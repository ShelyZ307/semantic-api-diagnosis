#!/usr/bin/env python3
"""Generate Stage 7 comparison report for weighted RoBERTa and real LLM baselines."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path


MAIN_VIEWS = ["seen_test", "unseen_family_test", "contract_dependent_unseen_test"]
MODEL_DIRS = {
    "DistilBERT": "outputs/distilbert_v1_seed42/evaluation",
    "original RoBERTa": "outputs/roberta_v1_seed42/evaluation",
    "weighted-loss RoBERTa": "outputs/roberta_weighted_v1_seed42/evaluation",
}
LLM_RESULTS = {
    "zero-shot LLM": {
        "seen_test": "data/generated/stage7_llm/zero_shot_seen_results.json",
        "unseen_family_test": "data/generated/stage7_llm/zero_shot_unseen_results.json",
        "contract_dependent_unseen_test": "data/generated/stage7_llm/zero_shot_contract_dependent_unseen_results.json",
    },
    "few-shot LLM": {
        "seen_test": "data/generated/stage7_llm/few_shot_seen_results.json",
        "unseen_family_test": "data/generated/stage7_llm/few_shot_unseen_results.json",
        "contract_dependent_unseen_test": "data/generated/stage7_llm/few_shot_contract_dependent_unseen_results.json",
    },
}


def main() -> None:
    parser = ArgumentParser(description="Generate the Stage 7 model comparison report.")
    parser.add_argument("--baseline-results", default="docs/results/baseline_results.json")
    parser.add_argument("--output", default="docs/results/stage_7_llm_and_weighted_roberta_results.md")
    args = parser.parse_args()
    baseline = _read_json(Path(args.baseline_results))
    models = {name: _read_json(Path(path) / "summary.json") for name, path in MODEL_DIRS.items()}
    llms = {
        name: {view: _optional_llm_metrics(Path(path)) for view, path in paths.items()}
        for name, paths in LLM_RESULTS.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(baseline, models, llms), encoding="utf-8")
    print(f"Wrote Stage 7 results report to {output}")


def _render(baseline: dict, models: dict, llms: dict) -> str:
    lines = [
        "# Stage 7: LLM Baselines, Weighted RoBERTa, And Unseen Error Analysis",
        "",
        "## Experiment Setup",
        "",
        "- Dataset, endpoint families, label taxonomy, seed, and validation-only threshold policy are unchanged.",
        "- Weighted RoBERTa is exactly one controlled improvement attempt using training-split `negatives / positives` weights in multi-label BCE.",
        "- Real LLM rows are reported only when provider-backed artifacts exist. Mock outputs are excluded.",
        "",
        "## Main Comparison",
        "",
        "| view | model | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate | semantic cross-field F1 | semantic domain F1 | semantic state F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in MAIN_VIEWS:
        lines.append(_row(view, "visible rule baseline", _baseline_metrics(baseline, view)))
        for name, summary in models.items():
            lines.append(_row(view, name, summary[view]["tuned"]))
        for name, results in llms.items():
            lines.append(_row(view, name, results[view]))
    lines.extend(
        [
            "",
            "## Weighted-Loss Delta",
            "",
            "| view | metric | original RoBERTa | weighted RoBERTa | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for view in ["seen_test", "unseen_family_test", "contract_dependent_seen_test", "contract_dependent_unseen_test"]:
        original = models["original RoBERTa"][view]["tuned"]
        weighted = models["weighted-loss RoBERTa"][view]["tuned"]
        for label, getter in [
            ("micro-F1", lambda x: x["all_labels"]["micro_f1"]),
            ("micro precision", lambda x: x["all_labels"]["micro_precision"]),
            ("semantic macro-F1", lambda x: x["semantic_labels"]["macro_f1"]),
            ("semantic domain F1", lambda x: x["semantic_labels"]["per_label"]["semantic_domain_constraint_violation"]["f1"]),
            ("exact match", lambda x: x["exact_match_label_set_accuracy"]),
            ("critical miss rate", lambda x: x["critical_semantic_error_miss_rate"]),
        ]:
            left, right = getter(original), getter(weighted)
            lines.append(f"| {view} | {label} | {left:.3f} | {right:.3f} | {right - left:+.3f} |")
    lines.extend(
        [
            "",
            "## Shortcut Ablation",
            "",
            "| model | split | full semantic macro-F1 | request-only | no constraints | request-only delta | no-constraints delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["original RoBERTa", "weighted-loss RoBERTa"]:
        summary = models[name]
        for split in ["seen_test", "unseen_family_test"]:
            full = summary[split]["tuned"]["semantic_labels"]["macro_f1"]
            request = summary[f"{split}_request_only"]["tuned"]["semantic_labels"]["macro_f1"]
            no_constraints = summary[f"{split}_no_constraints"]["tuned"]["semantic_labels"]["macro_f1"]
            lines.append(
                f"| {name} | {split} | {full:.3f} | {request:.3f} | {no_constraints:.3f} | "
                f"{request - full:+.3f} | {no_constraints - full:+.3f} |"
            )
    lines.extend(["", "## Real LLM Baseline Status", ""])
    if any(metrics is not None for results in llms.values() for metrics in results.values()):
        lines.append("- Provider-backed LLM sample artifacts are included in the main table.")
    else:
        lines.append(
            "- Blocked: no provider-backed LLM artifacts exist because no API credential was available in the workspace environment. "
            "The OpenAI Responses API path is implemented, but mock results remain excluded from scientific comparison."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpret(models),
            "",
            "See `docs/results/stage_7_unseen_error_analysis.md` for grouped failure analysis and concrete examples.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpret(models: dict) -> str:
    seen = models["original RoBERTa"]["seen_test"]["tuned"]
    original = models["original RoBERTa"]["unseen_family_test"]["tuned"]
    weighted = models["weighted-loss RoBERTa"]["unseen_family_test"]["tuned"]
    semantic_delta = weighted["semantic_labels"]["macro_f1"] - original["semantic_labels"]["macro_f1"]
    domain_delta = (
        weighted["semantic_labels"]["per_label"]["semantic_domain_constraint_violation"]["f1"]
        - original["semantic_labels"]["per_label"]["semantic_domain_constraint_violation"]["f1"]
    )
    miss_delta = weighted["critical_semantic_error_miss_rate"] - original["critical_semantic_error_miss_rate"]
    return (
        f"- Original RoBERTa beats the visible rule baseline on seen full-input micro-F1 (`{seen['all_labels']['micro_f1']:.3f}` "
        "vs `0.804`) but not on unseen semantic diagnosis (`0.352` vs `0.957` semantic macro-F1).\n"
        f"- Weighted loss changes unseen semantic macro-F1 by `{semantic_delta:+.3f}`, unseen semantic-domain F1 by "
        f"`{domain_delta:+.3f}`, and critical semantic miss rate by `{miss_delta:+.3f}`. It is rejected: the zero miss rate "
        "comes from broad overprediction, not better diagnosis.\n"
        "- Original RoBERTa is contract-sensitive: its semantic macro-F1 falls sharply on request-only and no-constraints views. "
        "Weighted RoBERTa loses that useful ablation pattern because its predictions are over-broad.\n"
        "- Real zero-shot and few-shot LLM comparisons remain blocked until a provider credential is available. No claim about "
        "instruction-following LLM performance is supported yet.\n"
        "- The defensible Stage 7 conclusion is that fine-tuned RoBERTa learns contract-sensitive behavior on seen families, "
        "while unseen-family transfer remains weak and the visible rule baseline remains stronger on held-out domains."
    )


def _row(view: str, model: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"| {view} | {model} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    semantic = metrics["semantic_labels"]
    per_label = semantic["per_label"]
    return (
        f"| {view} | {model} | {metrics['all_labels']['micro_f1']:.3f} | {semantic['macro_f1']:.3f} | "
        f"{metrics['exact_match_label_set_accuracy']:.3f} | {_fmt(metrics['critical_semantic_error_miss_rate'])} | "
        f"{per_label['semantic_cross_field_violation']['f1']:.3f} | "
        f"{per_label['semantic_domain_constraint_violation']['f1']:.3f} | "
        f"{per_label['semantic_state_violation']['f1']:.3f} |"
    )


def _baseline_metrics(baseline: dict, view: str) -> dict:
    section = "splits" if view in baseline["splits"] else "hard_subsets"
    return baseline[section][view]["visible_rule_based"]


def _optional_llm_metrics(path: Path) -> dict | None:
    return _read_json(path)["metrics"] if path.exists() else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
