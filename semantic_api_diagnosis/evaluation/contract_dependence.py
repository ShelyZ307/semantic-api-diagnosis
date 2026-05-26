"""Contract-dependence diagnostics for frozen evaluation splits."""

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from semantic_api_diagnosis.baselines.visible_rule_based import VisibleRuleBasedBaseline
from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.labels import SEMANTIC_LABELS
from semantic_api_diagnosis.quality_gate import _request_only_input
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


REQUEST_OBVIOUS_LABELS = {
    "missing_authentication",
    "malformed_url",
    "wrong_http_method",
    "wrong_type",
    "unexpected_or_malformed_body_structure",
}

CONTRACT_DEPENDENT_LABELS = {
    "missing_required_field",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
}


def classify_example(example: dict) -> str:
    labels = set(example.get("target", {}).get("error_labels", []))
    if not labels:
        return "valid"
    label_sources = {_label_source(example, label) for label in labels}
    has_obvious = "request_obvious" in label_sources
    has_dependent = "contract_dependent" in label_sources
    if has_obvious and has_dependent:
        return "mixed"
    if has_dependent:
        return "contract_dependent"
    return "request_obvious"


def analyze_contract_dependence(
    train_path: str | Path,
    seen_test_path: str | Path,
    unseen_test_path: str | Path,
    output_path: str | Path,
) -> dict:
    train = read_jsonl(train_path)
    splits = {
        "seen_test": read_jsonl(seen_test_path),
        "unseen_family_test": read_jsonl(unseen_test_path),
    }
    baseline = VisibleRuleBasedBaseline().fit(train)
    analysis = {}
    for split_name, rows in splits.items():
        analysis[split_name] = _split_analysis(rows, baseline)
    warnings = _warnings(analysis)
    report = _markdown_report(analysis, warnings)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    write_jsonl(
        [example for example in splits["seen_test"] if classify_example(example) == "contract_dependent"],
        output.parent / "contract_dependent_seen_test.jsonl",
    )
    write_jsonl(
        [example for example in splits["unseen_family_test"] if classify_example(example) == "contract_dependent"],
        output.parent / "contract_dependent_unseen_test.jsonl",
    )
    return {"analysis": analysis, "warnings": warnings, "report": report}


def _split_analysis(rows: list[dict], baseline: VisibleRuleBasedBaseline) -> dict:
    buckets = defaultdict(list)
    for example in rows:
        buckets[classify_example(example)].append(example)
    label_bucket_counts = {
        label: Counter()
        for example in rows
        for label in example.get("target", {}).get("error_labels", [])
    }
    for example in rows:
        bucket = classify_example(example)
        for label in example.get("target", {}).get("error_labels", []):
            label_bucket_counts[label][bucket] += 1
    semantic_examples = [
        example
        for example in rows
        if set(example.get("target", {}).get("error_labels", [])) & SEMANTIC_LABELS
    ]
    semantic_bucket_counts = Counter(classify_example(example) for example in semantic_examples)
    full_metrics_by_bucket = _metrics_by_bucket(buckets, baseline)
    request_only_rows = [_request_only_example(example) for example in rows]
    full_metrics = evaluate_predictions(rows, baseline.predict(rows))
    request_only_metrics = evaluate_predictions(request_only_rows, baseline.predict(request_only_rows))
    return {
        "total": len(rows),
        "bucket_counts": {bucket: len(buckets.get(bucket, [])) for bucket in ["valid", "request_obvious", "contract_dependent", "mixed"]},
        "label_bucket_counts": {label: dict(counts) for label, counts in sorted(label_bucket_counts.items())},
        "semantic_bucket_counts": dict(semantic_bucket_counts),
        "semantic_total": len(semantic_examples),
        "visible_rule_by_bucket": full_metrics_by_bucket,
        "visible_rule_full": _summary_metrics(full_metrics),
        "visible_rule_request_only": _summary_metrics(request_only_metrics),
    }


def _metrics_by_bucket(buckets: dict[str, list[dict]], baseline: VisibleRuleBasedBaseline) -> dict:
    output = {}
    for bucket in ["request_obvious", "contract_dependent", "mixed", "valid"]:
        rows = buckets.get(bucket, [])
        if not rows:
            output[bucket] = None
            continue
        output[bucket] = _summary_metrics(evaluate_predictions(rows, baseline.predict(rows)))
    return output


def _summary_metrics(metrics: dict) -> dict:
    return {
        "micro_f1": metrics["all_labels"]["micro_f1"],
        "macro_f1": metrics["all_labels"]["macro_f1"],
        "semantic_macro_f1": metrics["semantic_labels"]["macro_f1"],
        "semantic_micro_f1": metrics["semantic_labels"]["micro_f1"],
        "critical_semantic_miss_rate": metrics["critical_semantic_error_miss_rate"],
        "exact_match": metrics["exact_match_label_set_accuracy"],
        "validity_accuracy": metrics["validity"]["accuracy"],
        "severity_accuracy": metrics["severity_bucket"]["accuracy"],
    }


def _request_only_example(example: dict) -> dict:
    view = deepcopy(example)
    view["serialized_input"] = _request_only_input(example)
    return view


def _label_source(example: dict, label: str) -> str:
    if label in REQUEST_OBVIOUS_LABELS:
        return "request_obvious"
    if label in CONTRACT_DEPENDENT_LABELS:
        return "contract_dependent"
    if label == "invalid_value_range":
        return "request_obvious" if _obvious_invalid_range(example) else "contract_dependent"
    if label == "semantic_cross_field_violation":
        return "request_obvious" if _obvious_cross_field(example) else "contract_dependent"
    return "contract_dependent"


def _obvious_invalid_range(example: dict) -> bool:
    body = example.get("request", {}).get("body")
    if not isinstance(body, dict):
        return False
    positive_name_parts = ["amount", "quantity", "guests", "capacity", "days", "credits", "stock", "payment"]
    for field, value in body.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            if any(part in field for part in positive_name_parts):
                return True
    return False


def _obvious_cross_field(example: dict) -> bool:
    body = example.get("request", {}).get("body")
    url = example.get("request", {}).get("url", "")
    if not isinstance(body, dict):
        return False
    for field, value in body.items():
        if field.startswith("body_") and field.endswith("_id"):
            plain = field.removeprefix("body_")
            if plain in body and body[plain] != value:
                return True
            if isinstance(value, str) and value not in url:
                return True
    for field, value in body.items():
        if field.startswith("current_"):
            target = "target_" + field.removeprefix("current_")
            if target in body and body[target] == value:
                return True
    for left, right in [
        ("start_date", "end_date"),
        ("requested_slot_start", "requested_slot_end"),
        ("currency", "expected_currency"),
        ("warehouse_region", "customer_region"),
        ("destination_country", "return_country"),
    ]:
        if left in body and right in body and body[left] != body[right]:
            return True
    return False


def _warnings(analysis: dict) -> list[str]:
    warnings = []
    for split, data in analysis.items():
        semantic_total = data["semantic_total"] or 1
        contract_semantic = data["semantic_bucket_counts"].get("contract_dependent", 0)
        contract_semantic += data["semantic_bucket_counts"].get("mixed", 0)
        contract_share = contract_semantic / semantic_total
        if contract_share < 0.30:
            warnings.append(f"{split}: fewer than 30% of semantic-error examples are contract-dependent or mixed")
        contract_metrics = data["visible_rule_by_bucket"].get("contract_dependent")
        if contract_metrics and contract_metrics["semantic_macro_f1"] > 0.80:
            warnings.append(f"{split}: visible_rule_based semantic macro-F1 exceeds 0.80 on contract-dependent examples")
        full = data["visible_rule_full"]["semantic_macro_f1"]
        request_only = data["visible_rule_request_only"]["semantic_macro_f1"]
        if abs(full - request_only) <= 0.05:
            warnings.append(f"{split}: request-only semantic macro-F1 is within 0.05 of full-input performance")
    return warnings


def _markdown_report(analysis: dict, warnings: list[str]) -> str:
    lines = ["# Contract Dependence Analysis", ""]
    lines.extend(["## Bucket Distribution", "", "| split | valid | request_obvious | contract_dependent | mixed |", "|---|---:|---:|---:|---:|"])
    for split, data in analysis.items():
        total = data["total"]
        counts = data["bucket_counts"]
        lines.append(
            f"| {split} | {_count_pct(counts['valid'], total)} | {_count_pct(counts['request_obvious'], total)} | "
            f"{_count_pct(counts['contract_dependent'], total)} | {_count_pct(counts['mixed'], total)} |"
        )
    lines.extend(["", "## Error Labels by Bucket", ""])
    for split, data in analysis.items():
        lines.extend([f"### {split}", "", "| label | request_obvious | contract_dependent | mixed |", "|---|---:|---:|---:|"])
        for label, counts in data["label_bucket_counts"].items():
            lines.append(
                f"| {label} | {counts.get('request_obvious', 0)} | {counts.get('contract_dependent', 0)} | {counts.get('mixed', 0)} |"
            )
        lines.append("")
    lines.extend(["## Semantic Examples", "", "| split | request_obvious | contract_dependent | mixed |", "|---|---:|---:|---:|"])
    for split, data in analysis.items():
        total = data["semantic_total"] or 1
        counts = data["semantic_bucket_counts"]
        lines.append(
            f"| {split} | {_count_pct(counts.get('request_obvious', 0), total)} | "
            f"{_count_pct(counts.get('contract_dependent', 0), total)} | {_count_pct(counts.get('mixed', 0), total)} |"
        )
    lines.extend(["", "## Visible Rule-Based Performance by Bucket", "", "| split | bucket | micro-F1 | semantic macro-F1 | critical semantic miss rate | exact match |", "|---|---|---:|---:|---:|---:|"])
    for split, data in analysis.items():
        for bucket, metrics in data["visible_rule_by_bucket"].items():
            if metrics is None:
                lines.append(f"| {split} | {bucket} | n/a | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {split} | {bucket} | {metrics['micro_f1']:.3f} | {metrics['semantic_macro_f1']:.3f} | "
                    f"{_fmt(metrics['critical_semantic_miss_rate'])} | {metrics['exact_match']:.3f} |"
                )
    lines.extend(["", "## Full vs Request-Only Visible Baseline", "", "| split | view | micro-F1 | semantic macro-F1 | critical semantic miss rate |", "|---|---|---:|---:|---:|"])
    for split, data in analysis.items():
        for view_name in ["visible_rule_full", "visible_rule_request_only"]:
            metrics = data[view_name]
            lines.append(
                f"| {split} | {view_name} | {metrics['micro_f1']:.3f} | {metrics['semantic_macro_f1']:.3f} | "
                f"{_fmt(metrics['critical_semantic_miss_rate'])} |"
            )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    return "\n".join(lines) + "\n"


def _count_pct(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else "0 (0.0%)"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
