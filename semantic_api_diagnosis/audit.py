"""Quality audit reporting for pilot datasets."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, SEMANTIC_LABELS, STRUCTURAL_LABELS
from semantic_api_diagnosis.serialization.jsonl import read_jsonl

REQUIRED_INPUT_SECTIONS = [
    "Endpoint description",
    "Required constraints",
    "Request",
    "Method",
    "URL",
    "Authentication",
    "Body fields",
]

FORBIDDEN_SERIALIZED_TERMS = [
    "hidden_validate",
    "validate_example",
    "assert_quality",
    "FIELD_SEVERITIES",
    "def ",
    "class ",
]


def audit_dataset_file(
    dataset_path: str | Path,
    output_path: str | Path = "data/generated/pilot_audit_report.txt",
    samples_per_group: int = 2,
    show_full_input: bool = False,
) -> tuple[str, list[str]]:
    examples = read_jsonl(dataset_path)
    report, warnings = build_audit_report(examples, samples_per_group, show_full_input)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report, warnings


def build_audit_report(
    examples: list[dict],
    samples_per_group: int = 2,
    show_full_input: bool = False,
) -> tuple[str, list[str]]:
    warnings = collect_warnings(examples)
    lines = []
    lines.extend(_basic_counts(examples))
    lines.extend(_label_analysis(examples))
    lines.extend(_semantic_analysis(examples))
    lines.extend(_input_quality_checks(examples, warnings))
    lines.extend(_consistency_section(warnings))
    lines.extend(_diversity_checks(examples))
    lines.extend(_sample_section(examples, samples_per_group, show_full_input))
    return "\n".join(lines) + "\n", warnings


def collect_warnings(examples: list[dict]) -> list[str]:
    warnings = []
    for example in examples:
        example_id = example.get("id", "<missing-id>")
        target = example.get("target", {})
        labels = target.get("error_labels", [])
        validity = target.get("validity")
        severity = target.get("severity_bucket")
        metadata = example.get("generation_metadata", {})
        serialized = example.get("serialized_input")

        if not serialized:
            warnings.append(f"{example_id}: serialized_input is missing or empty")
        elif any(term in serialized for term in FORBIDDEN_SERIALIZED_TERMS):
            warnings.append(f"{example_id}: serialized_input may expose hidden validator code")
        if serialized:
            for section in REQUIRED_INPUT_SECTIONS:
                if section not in serialized:
                    warnings.append(f"{example_id}: serialized_input missing section '{section}'")

        if any(label not in FIXED_LABEL_TAXONOMY for label in labels):
            warnings.append(f"{example_id}: target contains label outside fixed taxonomy")
        if validity == "valid" and labels:
            warnings.append(f"{example_id}: valid example has error labels")
        if validity == "valid" and severity != "none":
            warnings.append(f"{example_id}: valid example has non-none severity_bucket")
        if validity == "invalid" and not labels:
            warnings.append(f"{example_id}: invalid example has no error labels")
        if validity == "invalid" and severity == "none":
            warnings.append(f"{example_id}: invalid example has severity_bucket none")
        if metadata.get("error_complexity") == "multi_error" and len(labels) < 2:
            warnings.append(f"{example_id}: multi_error example has fewer than two labels")
        if metadata.get("num_errors") != len(labels):
            warnings.append(f"{example_id}: generation_metadata.num_errors does not match labels")
    warnings.extend(_distribution_warnings(examples))
    return warnings


def _distribution_warnings(examples: list[dict]) -> list[str]:
    if len(examples) < 100:
        return []

    warnings = []
    label_counts = _label_counter(examples)
    for label in sorted(FIXED_LABEL_TAXONOMY):
        if label_counts[label] < 5:
            warnings.append(f"dataset: label '{label}' appears fewer than 5 times")
    for label in sorted(SEMANTIC_LABELS):
        if label_counts[label] < 5:
            warnings.append(f"dataset: semantic label '{label}' appears fewer than 5 times")

    multi_examples = [
        example
        for example in examples
        if len(example.get("target", {}).get("error_labels", [])) > 1
    ]
    if multi_examples:
        pair_counts = Counter()
        for example in multi_examples:
            labels = sorted(set(example.get("target", {}).get("error_labels", [])))
            for left, right in combinations(labels, 2):
                pair_counts[f"{left} + {right}"] += 1
        threshold = len(multi_examples) * 0.30
        for pair, count in pair_counts.items():
            if count > threshold:
                warnings.append(
                    f"dataset: label pair '{pair}' appears in more than 30% of multi-error examples"
                )
    return warnings


def _basic_counts(examples: list[dict]) -> list[str]:
    total = len(examples)
    validity = Counter(example.get("target", {}).get("validity", "missing") for example in examples)
    lines = ["# Pilot Dataset Audit", "", "## Basic Counts", f"total examples: {total}"]
    for value in ["valid", "invalid", "missing"]:
        count = validity.get(value, 0)
        if count:
            lines.append(f"{value}: {count} ({_pct(count, total)})")
    lines.extend(
        [
            "",
            "endpoint_family distribution:",
            *_counter_lines(Counter(example.get("endpoint_family", "missing") for example in examples)),
            "",
            "error_complexity distribution:",
            *_counter_lines(Counter(example.get("generation_metadata", {}).get("error_complexity", "missing") for example in examples)),
            "",
            "severity_bucket distribution:",
            *_counter_lines(Counter(example.get("target", {}).get("severity_bucket", "missing") for example in examples)),
            "",
        ]
    )
    return lines


def _label_analysis(examples: list[dict]) -> list[str]:
    label_counts = _label_counter(examples)
    label_lengths = Counter(len(example.get("target", {}).get("error_labels", [])) for example in examples)
    structural = sum(label_counts[label] for label in STRUCTURAL_LABELS)
    semantic = sum(label_counts[label] for label in SEMANTIC_LABELS)
    lines = [
        "## Label Analysis",
        "count per error label:",
        *_counter_lines(label_counts),
        f"structural label count: {structural}",
        f"semantic label count: {semantic}",
        f"examples with no labels: {label_lengths.get(0, 0)}",
        f"examples with one label: {label_lengths.get(1, 0)}",
        f"examples with multiple labels: {sum(count for size, count in label_lengths.items() if size > 1)}",
        "",
        "label co-occurrence table:",
    ]
    co_occurrence = Counter()
    for example in examples:
        labels = sorted(set(example.get("target", {}).get("error_labels", [])))
        for left, right in combinations(labels, 2):
            co_occurrence[f"{left} + {right}"] += 1
    lines.extend(_counter_lines(co_occurrence) if co_occurrence else ["- none"])
    lines.append("")
    return lines


def _semantic_analysis(examples: list[dict]) -> list[str]:
    label_counts = _label_counter(examples)
    semantic_together = []
    semantic_with_structural = []
    for example in examples:
        labels = set(example.get("target", {}).get("error_labels", []))
        semantic_labels = labels & SEMANTIC_LABELS
        structural_labels = labels & STRUCTURAL_LABELS
        if len(semantic_labels) > 1:
            semantic_together.append(example["id"])
        if semantic_labels and structural_labels:
            semantic_with_structural.append(example["id"])
    return [
        "## Semantic-Specific Analysis",
        f"semantic_cross_field_violation: {label_counts['semantic_cross_field_violation']}",
        f"semantic_domain_constraint_violation: {label_counts['semantic_domain_constraint_violation']}",
        f"semantic_state_violation: {label_counts['semantic_state_violation']}",
        f"examples where semantic labels appear together: {_id_list(semantic_together)}",
        f"examples where semantic labels appear with structural labels: {_id_list(semantic_with_structural)}",
        "",
    ]


def _input_quality_checks(examples: list[dict], warnings: list[str]) -> list[str]:
    non_empty = sum(1 for example in examples if example.get("serialized_input"))
    forbidden = sum(
        1
        for example in examples
        if any(term in example.get("serialized_input", "") for term in FORBIDDEN_SERIALIZED_TERMS)
    )
    all_sections = sum(
        1
        for example in examples
        if all(section in example.get("serialized_input", "") for section in REQUIRED_INPUT_SECTIONS)
    )
    return [
        "## Input Quality Checks",
        f"serialized_input non-empty: {non_empty}/{len(examples)}",
        f"serialized_input exposing forbidden terms: {forbidden}",
        f"serialized_input containing all required sections: {all_sections}/{len(examples)}",
        f"input-quality warnings: {sum('serialized_input' in warning for warning in warnings)}",
        "",
    ]


def _consistency_section(warnings: list[str]) -> list[str]:
    lines = ["## Consistency Checks"]
    if warnings:
        lines.append(f"warnings: {len(warnings)}")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("warnings: 0")
    lines.append("")
    return lines


def _diversity_checks(examples: list[dict]) -> list[str]:
    descriptions = {
        example.get("endpoint_contract", {}).get("description", "")
        for example in examples
    }
    constraints = {
        constraint.get("constraint_text", "")
        for example in examples
        for constraint in example.get("endpoint_contract", {}).get("constraints", [])
    }
    urls = {example.get("request", {}).get("url", "") for example in examples}
    field_names = {
        field_name
        for example in examples
        for field_name in _field_names(example)
    }
    body_orders = {
        tuple(example.get("request", {}).get("body", {}).keys())
        for example in examples
        if isinstance(example.get("request", {}).get("body"), dict)
    }
    return [
        "## Diversity Checks",
        f"unique endpoint descriptions: {len(descriptions)}",
        f"unique constraint texts: {len(constraints)}",
        f"unique URLs or URL patterns: {len(urls)}",
        f"unique field names: {len(field_names)}",
        f"field order variants in body: {len(body_orders)}",
        "field order varies in serialized_input: yes" if len(body_orders) > 1 else "field order varies in serialized_input: no",
        "constraint wording varies: yes" if len(constraints) > 3 else "constraint wording varies: no",
        "",
    ]


def _sample_section(examples: list[dict], samples_per_group: int, show_full_input: bool) -> list[str]:
    lines = ["## Manual Review Samples"]
    by_family = defaultdict(lambda: defaultdict(list))
    by_semantic_label = defaultdict(list)
    for example in examples:
        by_family[example.get("endpoint_family")][example.get("target", {}).get("validity")].append(example)
        for label in example.get("target", {}).get("error_labels", []):
            if label in SEMANTIC_LABELS:
                by_semantic_label[label].append(example)

    lines.append("family samples:")
    for family in sorted(by_family):
        lines.append(f"### {family}")
        for validity in ["valid", "invalid"]:
            lines.append(f"{validity}:")
            lines.extend(_format_examples(by_family[family][validity][:samples_per_group], show_full_input))

    lines.append("semantic label samples:")
    for label in sorted(SEMANTIC_LABELS):
        lines.append(f"### {label}")
        lines.extend(_format_examples(by_semantic_label[label][:samples_per_group], show_full_input))
    return lines


def _format_examples(examples: list[dict], show_full_input: bool) -> list[str]:
    if not examples:
        return ["- none"]
    lines = []
    for example in examples:
        serialized = example.get("serialized_input", "")
        if not show_full_input and len(serialized) > 500:
            serialized = serialized[:500].rstrip() + "\n..."
        lines.extend(
            [
                f"- id: {example.get('id')}",
                f"  endpoint_family: {example.get('endpoint_family')}",
                f"  serialized_input:\n{_indent(serialized, '    ')}",
                f"  target: {example.get('target')}",
                f"  injected_errors: {example.get('generation_metadata', {}).get('injected_errors')}",
            ]
        )
    return lines


def _counter_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {counter[key]}" for key in sorted(counter)]


def _label_counter(examples: list[dict]) -> Counter:
    return Counter(
        label
        for example in examples
        for label in example.get("target", {}).get("error_labels", [])
    )


def _field_names(example: dict) -> list[str]:
    contract = example.get("endpoint_contract", {})
    body = example.get("request", {}).get("body", {})
    names = list(contract.get("required_fields", {})) + list(contract.get("optional_fields", {}))
    if isinstance(body, dict):
        names.extend(body)
    return names


def _pct(count: int, total: int) -> str:
    return "0.0%" if total == 0 else f"{count / total:.1%}"


def _id_list(ids: list[str]) -> str:
    return ", ".join(ids[:20]) if ids else "none"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
