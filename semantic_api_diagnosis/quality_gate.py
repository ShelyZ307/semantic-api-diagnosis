"""Stage 2.5 quality-gate utilities."""

from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path

from semantic_api_diagnosis.labels import SEMANTIC_LABELS
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl
from semantic_api_diagnosis.dataset_plan import (
    PLANNED_SEEN_ENDPOINT_FAMILIES,
    PLANNED_UNSEEN_ENDPOINT_FAMILIES,
)


def export_manual_review_sample(
    dataset_path: str | Path,
    output_path: str | Path,
    examples_per_family: int = 10,
) -> list[dict]:
    examples = read_jsonl(dataset_path)
    selected = []
    for family in sorted({example["endpoint_family"] for example in examples}):
        family_examples = [example for example in examples if example["endpoint_family"] == family]
        selected.extend(_stratified_family_sample(family_examples, examples_per_family))

    write_jsonl(selected, output_path)
    markdown_path = Path(output_path).with_suffix(".md")
    markdown_path.write_text(_manual_review_markdown(selected), encoding="utf-8")
    return selected


def duplicate_report(
    dataset_path: str | Path,
    output_path: str | Path = "data/generated/duplicate_report.txt",
    near_duplicate_threshold: float = 0.92,
    max_near_duplicate_pairs: int = 50,
    max_groups: int = 10,
    show_examples: bool = False,
) -> str:
    examples = read_jsonl(dataset_path)
    serialized_values = [example.get("serialized_input", "") for example in examples]
    body_values = [_canonical(example.get("request", {}).get("body")) for example in examples]
    contract_request_values = [
        _canonical(
            {
                "endpoint_contract": example.get("endpoint_contract"),
                "request": example.get("request"),
            }
        )
        for example in examples
    ]
    near_duplicates = _near_duplicate_pairs(
        examples,
        serialized_values,
        near_duplicate_threshold,
        max_near_duplicate_pairs,
    )
    duplicate_body_groups = _duplicate_groups(examples, body_values)
    serialized_duplicate_count = _duplicate_count(serialized_values)
    body_duplicate_count = _duplicate_count(body_values)
    contract_request_duplicate_count = _duplicate_count(contract_request_values)
    warnings = _duplicate_warnings(
        duplicate_body_groups,
        serialized_duplicate_count,
        contract_request_duplicate_count,
        len(near_duplicates),
    )
    lines = [
        "Duplicate Report",
        "",
        f"total examples: {len(examples)}",
        f"exact duplicate serialized_input count: {serialized_duplicate_count}",
        f"exact duplicate request body count: {body_duplicate_count}",
        f"exact duplicate endpoint contract + request count: {contract_request_duplicate_count}",
        f"near-duplicate serialized_input pairs >= {near_duplicate_threshold:.2f}: {len(near_duplicates)}",
        "",
        "warnings:",
        *([f"- {warning}" for warning in warnings] if warnings else ["- none"]),
    ]
    if duplicate_body_groups:
        lines.append("")
        lines.append(f"duplicate request-body groups (top {max_groups}):")
        for group_index, group in enumerate(duplicate_body_groups[:max_groups], start=1):
            lines.extend(_format_duplicate_body_group(group_index, group, show_examples))
    else:
        lines.extend(["", "duplicate request-body groups: none"])
    if near_duplicates:
        lines.append("")
        lines.append("near-duplicate pairs:")
        for left_id, right_id, score in near_duplicates:
            left = _example_by_id(examples, left_id)
            right = _example_by_id(examples, right_id)
            lines.extend(_format_near_duplicate_pair(left, right, score, show_examples))
    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def create_shortcut_views(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    examples = read_jsonl(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    view_builders = {
        "request_only": _request_only_input,
        "contract_only": _contract_only_input,
        "no_constraints": _no_constraints_input,
        "family_only": _family_only_input,
    }
    paths = {}
    for view_name, builder in view_builders.items():
        rows = []
        for example in examples:
            view = deepcopy(example)
            view["serialized_input"] = builder(example)
            rows.append(view)
        path = output / f"{view_name}.jsonl"
        write_jsonl(rows, path)
        paths[view_name] = path
    return paths


def seen_unseen_overlap_report(
    seen_path: str | Path,
    unseen_path: str | Path,
    output_path: str | Path = "data/generated/seen_unseen_overlap_report.txt",
    similarity_threshold: float = 0.78,
) -> str:
    seen_examples = read_jsonl(seen_path)
    unseen_examples = read_jsonl(unseen_path)
    seen_families = sorted({example["endpoint_family"] for example in seen_examples})
    unseen_families = sorted({example["endpoint_family"] for example in unseen_examples})
    seen_fields = _field_names(seen_examples)
    unseen_fields = _field_names(unseen_examples)
    overlapping_fields = sorted(seen_fields & unseen_fields)
    field_denominator = len(seen_fields | unseen_fields) or 1
    description_pairs = _similar_text_pairs(
        _family_texts(seen_examples, "description"),
        _family_texts(unseen_examples, "description"),
        similarity_threshold,
    )
    constraint_pairs = _similar_text_pairs(
        _family_constraint_texts(seen_examples),
        _family_constraint_texts(unseen_examples),
        similarity_threshold,
    )
    suspicious_pairs = [
        ("refunds_orders", "shipping_returns"),
        ("booking_reservation", "course_registration"),
        ("user_permissions", "subscription_plan_changes"),
    ]
    lines = [
        "Seen vs Unseen Overlap Report",
        "",
        f"seen dataset: {seen_path}",
        f"unseen dataset: {unseen_path}",
        "",
        "Family Sets",
        f"- seen families: {seen_families}",
        f"- unseen families: {unseen_families}",
        f"- seen and unseen family sets disjoint: {set(seen_families).isdisjoint(unseen_families)}",
        "",
        "Field Name Overlap",
        f"- seen field count: {len(seen_fields)}",
        f"- unseen field count: {len(unseen_fields)}",
        f"- exact overlapping field names: {overlapping_fields if overlapping_fields else 'none'}",
        f"- field overlap percentage: {len(overlapping_fields) / field_denominator:.1%}",
        "",
        "URL Path Patterns",
        f"- seen patterns: {sorted(_url_patterns(seen_examples))}",
        f"- unseen patterns: {sorted(_url_patterns(unseen_examples))}",
        "",
        "Status / Enum-like Values",
        f"- seen values: {sorted(_enum_like_values(seen_examples))}",
        f"- unseen values: {sorted(_enum_like_values(unseen_examples))}",
        f"- overlapping values: {sorted(_enum_like_values(seen_examples) & _enum_like_values(unseen_examples))}",
        "",
        "Semantic Rule Templates",
        f"- seen templates: {sorted(_semantic_rule_templates(seen_examples))}",
        f"- unseen templates: {sorted(_semantic_rule_templates(unseen_examples))}",
        f"- overlapping templates: {sorted(_semantic_rule_templates(seen_examples) & _semantic_rule_templates(unseen_examples))}",
        "",
        f"Similar Endpoint Description Pairs >= {similarity_threshold:.2f}",
        *(_format_text_pairs(description_pairs) or ["- none"]),
        "",
        f"Similar Constraint Text Pairs >= {similarity_threshold:.2f}",
        *(_format_text_pairs(constraint_pairs[:20]) or ["- none"]),
        "",
        "Suspicious Family-Pair Similarities",
    ]
    for seen_family, unseen_family in suspicious_pairs:
        lines.extend(_format_family_similarity(seen_examples, unseen_examples, seen_family, unseen_family))
    lines.extend(_known_unseen_near_duplicate_inspection(unseen_examples))
    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def cross_split_leakage_report(
    split_paths: list[str | Path],
    output_path: str | Path = "data/generated/cross_split_leakage_report.txt",
    near_duplicate_threshold: float = 0.92,
    max_pairs: int | None = None,
) -> str:
    tagged = []
    for path in split_paths:
        for example in read_jsonl(path):
            tagged.append({"path": str(path), "example": example})
    serialized_duplicates = _cross_duplicate_pairs(tagged, lambda item: item["example"].get("serialized_input", ""))
    contract_request_duplicates = _cross_duplicate_pairs(
        tagged,
        lambda item: _canonical(
            {
                "endpoint_contract": item["example"].get("endpoint_contract"),
                "request": item["example"].get("request"),
            }
        ),
    )
    near_pairs = _cross_near_duplicate_pairs(tagged, near_duplicate_threshold, max_pairs)
    family_leaks = _family_leakage_findings(tagged)
    suspicious_same_target = [
        pair
        for pair in near_pairs
        if pair[0]["example"].get("target") == pair[1]["example"].get("target")
    ]
    near_by_split_pair = Counter(
        tuple(sorted([Path(left["path"]).name, Path(right["path"]).name]))
        for left, right, _score in near_pairs
    )
    near_by_family = Counter(
        left["example"]["endpoint_family"]
        if left["example"]["endpoint_family"] == right["example"]["endpoint_family"]
        else f"{left['example']['endpoint_family']} + {right['example']['endpoint_family']}"
        for left, right, _score in near_pairs
    )
    strict_pass = (
        not serialized_duplicates
        and not contract_request_duplicates
        and not near_pairs
        and not suspicious_same_target
        and not family_leaks
    )
    lines = [
        "Cross-Split Leakage Report",
        "",
        f"split files: {[str(path) for path in split_paths]}",
        f"total examples compared: {len(tagged)}",
        "",
        f"exact duplicate serialized_input across files: {len(serialized_duplicates)}",
        *(_format_cross_pairs(serialized_duplicates[:20]) if serialized_duplicates else ["- none"]),
        "",
        f"exact duplicate endpoint_contract + request across files: {len(contract_request_duplicates)}",
        *(_format_cross_pairs(contract_request_duplicates[:20]) if contract_request_duplicates else ["- none"]),
        "",
        f"near-duplicate serialized_input pairs across files >= {near_duplicate_threshold:.2f}: {len(near_pairs)}",
        *(_format_cross_near_pairs(near_pairs[:20]) if near_pairs else ["- none"]),
        "",
        "near-duplicates by split pair:",
        *(_counter_item_lines(near_by_split_pair) if near_by_split_pair else ["- none"]),
        "",
        "near-duplicates by endpoint family:",
        *(_counter_item_lines(near_by_family) if near_by_family else ["- none"]),
        "",
        "family leakage findings:",
        *([f"- {finding}" for finding in family_leaks] if family_leaks else ["- none"]),
        "",
        f"total suspicious same-target high-similarity pairs: {len(suspicious_same_target)}",
        "suspicious same-target high-similarity pairs:",
        *(_format_cross_near_pairs(suspicious_same_target[:20]) if suspicious_same_target else ["- none"]),
        "",
        "strict leakage criteria:",
        f"- exact duplicate serialized_input across splits == 0: {len(serialized_duplicates) == 0}",
        f"- exact duplicate endpoint_contract + request across splits == 0: {len(contract_request_duplicates) == 0}",
        f"- near-duplicate serialized_input pairs across splits >= {near_duplicate_threshold:.2f} == 0: {len(near_pairs) == 0}",
        f"- suspicious same-target high-similarity pairs == 0: {len(suspicious_same_target) == 0}",
        f"- family leakage findings == none: {not family_leaks}",
        f"strict leakage pass: {strict_pass}",
    ]
    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def serialized_input_similarity(left: str, right: str) -> float:
    return _sequence_similarity(_normalize_text(left), _normalize_text(right))


def is_near_duplicate_serialized_input(left: str, right: str, threshold: float = 0.92) -> bool:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if _token_overlap(set(left_normalized.split()), set(right_normalized.split())) < threshold - 0.08:
        return False
    return _sequence_similarity(left_normalized, right_normalized) >= threshold


def _field_names(examples: list[dict]) -> set[str]:
    fields = set()
    for example in examples:
        contract = example.get("endpoint_contract", {})
        fields.update(contract.get("required_fields", {}).keys())
        fields.update(contract.get("optional_fields", {}).keys())
        body = example.get("request", {}).get("body")
        if isinstance(body, dict):
            fields.update(body.keys())
    return fields


def _url_patterns(examples: list[dict]) -> set[str]:
    patterns = set()
    for example in examples:
        template = example.get("endpoint_contract", {}).get("url_template")
        if template:
            patterns.add(template)
    return patterns


def _enum_like_values(examples: list[dict]) -> set[str]:
    values = set()
    for example in examples:
        required = example.get("endpoint_contract", {}).get("required_fields", {})
        optional = example.get("endpoint_contract", {}).get("optional_fields", {})
        for type_spec in list(required.values()) + list(optional.values()):
            if isinstance(type_spec, str) and type_spec.startswith("enum:"):
                values.update(type_spec.removeprefix("enum:").split("|"))
        body = example.get("request", {}).get("body")
        if isinstance(body, dict):
            for field_name, value in body.items():
                if isinstance(value, str) and any(marker in field_name for marker in ["status", "role", "plan", "priority"]):
                    values.add(value)
    return values


def _semantic_rule_templates(examples: list[dict]) -> set[str]:
    templates = set()
    for example in examples:
        constraints = example.get("endpoint_contract", {}).get("constraints", [])
        for constraint in constraints:
            constraint_type = constraint.get("constraint_type", "")
            if not constraint_type.startswith("semantic"):
                continue
            fields = constraint.get("fields", [])
            templates.add(f"{constraint_type}:fields={len(fields)}")
    return templates


def _family_texts(examples: list[dict], key: str) -> dict[str, set[str]]:
    values = defaultdict(set)
    for example in examples:
        contract = example.get("endpoint_contract", {})
        text = contract.get(key, "")
        if text:
            values[example["endpoint_family"]].add(text)
    return values


def _family_constraint_texts(examples: list[dict]) -> dict[str, set[str]]:
    values = defaultdict(set)
    for example in examples:
        for constraint in example.get("endpoint_contract", {}).get("constraints", []):
            text = constraint.get("constraint_text", "")
            if text:
                values[example["endpoint_family"]].add(text)
    return values


def _similar_text_pairs(
    left_by_family: dict[str, set[str]],
    right_by_family: dict[str, set[str]],
    threshold: float,
) -> list[tuple[str, str, str, str, float]]:
    pairs = []
    for left_family, left_values in left_by_family.items():
        for right_family, right_values in right_by_family.items():
            for left_text in left_values:
                for right_text in right_values:
                    score = _sequence_similarity(
                        _normalize_text(left_text),
                        _normalize_text(right_text),
                    )
                    if score >= threshold:
                        pairs.append((left_family, right_family, left_text, right_text, score))
    return sorted(pairs, key=lambda pair: pair[4], reverse=True)


def _format_text_pairs(pairs: list[tuple[str, str, str, str, float]]) -> list[str]:
    return [
        f"- {left_family} vs {right_family}: {score:.3f} | {_snippet(left_text, 90)} <> {_snippet(right_text, 90)}"
        for left_family, right_family, left_text, right_text, score in pairs
    ]


def _format_family_similarity(
    seen_examples: list[dict],
    unseen_examples: list[dict],
    seen_family: str,
    unseen_family: str,
) -> list[str]:
    seen_family_examples = [example for example in seen_examples if example["endpoint_family"] == seen_family]
    unseen_family_examples = [example for example in unseen_examples if example["endpoint_family"] == unseen_family]
    seen_fields = _field_names(seen_family_examples)
    unseen_fields = _field_names(unseen_family_examples)
    field_union = len(seen_fields | unseen_fields) or 1
    description_pairs = _similar_text_pairs(
        _family_texts(seen_family_examples, "description"),
        _family_texts(unseen_family_examples, "description"),
        0.0,
    )
    constraint_pairs = _similar_text_pairs(
        _family_constraint_texts(seen_family_examples),
        _family_constraint_texts(unseen_family_examples),
        0.0,
    )
    max_description = description_pairs[0][4] if description_pairs else 0.0
    max_constraint = constraint_pairs[0][4] if constraint_pairs else 0.0
    return [
        f"- {seen_family} vs {unseen_family}",
        f"  field_overlap: {len(seen_fields & unseen_fields) / field_union:.1%}",
        f"  overlapping_fields: {sorted(seen_fields & unseen_fields) if seen_fields & unseen_fields else 'none'}",
        f"  max_description_similarity: {max_description:.3f}",
        f"  max_constraint_similarity: {max_constraint:.3f}",
    ]


def _known_unseen_near_duplicate_inspection(unseen_examples: list[dict]) -> list[str]:
    examples_by_id = {example["id"]: example for example in unseen_examples}
    if "ex_000359" not in examples_by_id or "ex_000387" not in examples_by_id:
        return [
            "",
            "Known Unseen Near-Duplicate Inspection",
            "- ex_000359 + ex_000387: not present in this unseen dataset",
        ]
    left = examples_by_id["ex_000359"]
    right = examples_by_id["ex_000387"]
    score = _sequence_similarity(
        _normalize_text(left.get("serialized_input", "")),
        _normalize_text(right.get("serialized_input", "")),
    )
    return [
        "",
        "Known Unseen Near-Duplicate Inspection",
        f"- examples: ex_000359 + ex_000387",
        f"  similarity_score: {score:.3f}",
        f"  endpoint_families: {[left['endpoint_family'], right['endpoint_family']]}",
        f"  targets: {[left['target']['validity'], right['target']['validity']]}",
        f"  error_labels: {[left['target']['error_labels'], right['target']['error_labels']]}",
        f"  url_differs: {left.get('request', {}).get('url') != right.get('request', {}).get('url')}",
        f"  body_differs: {left.get('request', {}).get('body') != right.get('request', {}).get('body')}",
        f"  target_differs: {left.get('target') != right.get('target')}",
        "  inspected_assessment: reported for manual inspection; not an exact duplicate or cross-split leak",
    ]


def _cross_duplicate_pairs(tagged: list[dict], key_fn) -> list[tuple[dict, dict]]:
    grouped = defaultdict(list)
    for item in tagged:
        grouped[key_fn(item)].append(item)
    pairs = []
    for group in grouped.values():
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                if group[left_index]["path"] != group[right_index]["path"]:
                    pairs.append((group[left_index], group[right_index]))
    return pairs


def _cross_near_duplicate_pairs(
    tagged: list[dict],
    threshold: float,
    max_pairs: int | None,
) -> list[tuple[dict, dict, float]]:
    normalized = [
        _normalize_text(item["example"].get("serialized_input", ""))
        for item in tagged
    ]
    token_sets = [set(value.split()) for value in normalized]
    pairs = []
    for left_index in range(len(tagged)):
        for right_index in range(left_index + 1, len(tagged)):
            if tagged[left_index]["path"] == tagged[right_index]["path"]:
                continue
            overlap = _token_overlap(token_sets[left_index], token_sets[right_index])
            if overlap < threshold - 0.08:
                continue
            score = _sequence_similarity(normalized[left_index], normalized[right_index])
            if score >= threshold:
                pairs.append((tagged[left_index], tagged[right_index], score))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _family_leakage_findings(tagged: list[dict]) -> list[str]:
    findings = []
    seen = set(PLANNED_SEEN_ENDPOINT_FAMILIES)
    unseen = set(PLANNED_UNSEEN_ENDPOINT_FAMILIES)
    for item in tagged:
        path = item["path"]
        family = item["example"]["endpoint_family"]
        path_name = Path(path).name.lower()
        if "unseen" in path_name and family in seen:
            findings.append(f"seen family {family} appears in unseen-like file {path}")
        if "seen" in path_name and "unseen" not in path_name and family in unseen:
            findings.append(f"unseen family {family} appears in seen-like file {path}")
    return findings


def _format_cross_pairs(pairs: list[tuple[dict, dict]]) -> list[str]:
    return [
        (
            f"- {left['path']}:{left['example']['id']} + "
            f"{right['path']}:{right['example']['id']} "
            f"families={[left['example']['endpoint_family'], right['example']['endpoint_family']]}"
        )
        for left, right in pairs
    ]


def _format_cross_near_pairs(pairs: list[tuple[dict, dict, float]]) -> list[str]:
    return [
        (
            f"- {left['path']}:{left['example']['id']} + "
            f"{right['path']}:{right['example']['id']} "
            f"score={score:.3f} "
            f"families={[left['example']['endpoint_family'], right['example']['endpoint_family']]} "
            f"same_target={left['example'].get('target') == right['example'].get('target')}"
        )
        for left, right, score in pairs
    ]


def _counter_item_lines(counter: Counter) -> list[str]:
    return [
        f"- {key}: {count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def _stratified_family_sample(examples: list[dict], count: int) -> list[dict]:
    buckets = {
        "valid": [example for example in examples if example["target"]["validity"] == "valid"],
        "single_error": [
            example
            for example in examples
            if example["generation_metadata"]["error_complexity"] == "single_error"
        ],
        "multi_error": [
            example
            for example in examples
            if example["generation_metadata"]["error_complexity"] == "multi_error"
        ],
        "semantic": [
            example
            for example in examples
            if set(example["target"]["error_labels"]) & SEMANTIC_LABELS
        ],
    }
    selected = []
    seen_ids = set()
    bucket_order = ["valid", "single_error", "multi_error", "semantic"]
    while len(selected) < count:
        made_progress = False
        for bucket_name in bucket_order:
            for example in buckets[bucket_name]:
                if example["id"] in seen_ids:
                    continue
                selected.append(example)
                seen_ids.add(example["id"])
                made_progress = True
                break
            if len(selected) == count:
                break
        if not made_progress:
            break
    if len(selected) < count:
        for example in examples:
            if example["id"] not in seen_ids:
                selected.append(example)
                seen_ids.add(example["id"])
            if len(selected) == count:
                break
    return selected


def _manual_review_markdown(examples: list[dict]) -> str:
    lines = ["# Manual Review Sample", ""]
    for example in examples:
        lines.extend(
            [
                f"## {example['id']}",
                f"endpoint_family: {example['endpoint_family']}",
                "",
                "serialized_input:",
                "```text",
                example["serialized_input"],
                "```",
                "",
                f"target: {example['target']}",
                f"injected_errors: {example['generation_metadata']['injected_errors']}",
                "",
                "reviewer_label_correct: yes/no/unsure",
                "missing_labels:",
                "notes:",
                "",
            ]
        )
    return "\n".join(lines)


def _request_only_input(example: dict) -> str:
    request = example["request"]
    body = request.get("body")
    body_fields = _format_mapping(body) if isinstance(body, dict) else f"- body: {body}"
    return (
        f"Method: {request['method']}\n"
        f"URL: {request['url']}\n"
        f"Authentication: {request['authentication']}\n"
        "Query parameters:\n"
        f"{_format_mapping(request.get('query_params', {}))}\n"
        "Body fields:\n"
        f"{body_fields}"
    )


def _contract_only_input(example: dict) -> str:
    contract = example["endpoint_contract"]
    constraints = "\n".join(
        f"- {constraint['constraint_text']}"
        for constraint in contract["constraints"]
    )
    return (
        "Endpoint description:\n"
        f"{contract['description']}\n\n"
        "Required constraints:\n"
        f"{constraints}"
    )


def _no_constraints_input(example: dict) -> str:
    contract = example["endpoint_contract"]
    return (
        "Endpoint description:\n"
        f"{contract['description']}\n\n"
        "Request:\n"
        f"{_request_only_input(example)}"
    )


def _family_only_input(example: dict) -> str:
    return f"endpoint_family: {example['endpoint_family']}"


def _format_mapping(values: dict) -> str:
    if not values:
        return "none"
    return "\n".join(f"- {name}: {value}" for name, value in values.items())


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _duplicate_groups(examples: list[dict], keys: list[str]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for example, key in zip(examples, keys):
        grouped[key].append(example)
    groups = [group for group in grouped.values() if len(group) > 1]
    return sorted(groups, key=lambda group: (-len(group), [example["id"] for example in group]))


def _duplicate_warnings(
    duplicate_body_groups: list[list[dict]],
    serialized_duplicate_count: int,
    contract_request_duplicate_count: int,
    near_duplicate_count: int,
) -> list[str]:
    warnings = []
    if serialized_duplicate_count > 0:
        warnings.append("exact duplicate serialized_input count is greater than 0")
    if contract_request_duplicate_count > 0:
        warnings.append("exact duplicate endpoint_contract + request count is greater than 0")
    if near_duplicate_count > 10:
        warnings.append("near-duplicate serialized_input pair count is greater than 10")
    for index, group in enumerate(duplicate_body_groups, start=1):
        pattern_counts = Counter(
            (
                example["endpoint_family"],
                tuple(example.get("target", {}).get("error_labels", [])),
            )
            for example in group
        )
        for (family, labels), count in pattern_counts.items():
            if count > 3:
                label_text = ", ".join(labels) if labels else "none"
                warnings.append(
                    f"duplicate request-body group {index} repeats family={family} labels={label_text} {count} times"
                )
    return warnings


def _format_duplicate_body_group(group_index: int, group: list[dict], show_examples: bool) -> list[str]:
    serialized_values = {example.get("serialized_input", "") for example in group}
    contract_request_values = {
        _canonical(
            {
                "endpoint_contract": example.get("endpoint_contract"),
                "request": example.get("request"),
            }
        )
        for example in group
    }
    lines = [
        f"- duplicate_group_id: body_group_{group_index:03d}",
        f"  num_examples: {len(group)}",
        f"  example_ids: {[example['id'] for example in group]}",
        f"  endpoint_families: {[example['endpoint_family'] for example in group]}",
        f"  targets: {[example['target']['validity'] for example in group]}",
        f"  error_labels: {[example['target']['error_labels'] for example in group]}",
        f"  serialized_input_identical: {len(serialized_values) == 1}",
        f"  endpoint_contract_plus_request_identical: {len(contract_request_values) == 1}",
    ]
    if show_examples:
        lines.append("  examples:")
        for example in group:
            lines.extend(
                [
                    f"    - id: {example['id']}",
                    f"      serialized_input_snippet: {_snippet(example.get('serialized_input', ''))}",
                ]
            )
    return lines


def _format_near_duplicate_pair(left: dict, right: dict, score: float, show_examples: bool) -> list[str]:
    left_labels = left.get("target", {}).get("error_labels", [])
    right_labels = right.get("target", {}).get("error_labels", [])
    lines = [
        f"- examples: {left['id']} + {right['id']}",
        f"  similarity_score: {score:.3f}",
        f"  endpoint_families: {[left['endpoint_family'], right['endpoint_family']]}",
        f"  targets: {[left['target']['validity'], right['target']['validity']]}",
        f"  error_labels: {[left_labels, right_labels]}",
        f"  same_endpoint_family: {left['endpoint_family'] == right['endpoint_family']}",
        f"  same_error_labels: {left_labels == right_labels}",
        "  difference_summary:",
        f"    url_differs: {left.get('request', {}).get('url') != right.get('request', {}).get('url')}",
        f"    body_differs: {left.get('request', {}).get('body') != right.get('request', {}).get('body')}",
        f"    constraints_differ: {_constraint_texts(left) != _constraint_texts(right)}",
        f"    target_differs: {left.get('target') != right.get('target')}",
    ]
    if show_examples:
        lines.extend(
            [
                "  serialized_input_snippets:",
                f"    {left['id']}: {_snippet(left.get('serialized_input', ''))}",
                f"    {right['id']}: {_snippet(right.get('serialized_input', ''))}",
            ]
        )
    return lines


def _example_by_id(examples: list[dict], example_id: str) -> dict:
    for example in examples:
        if example["id"] == example_id:
            return example
    raise KeyError(example_id)


def _constraint_texts(example: dict) -> list[str]:
    return [
        constraint.get("constraint_text", "")
        for constraint in example.get("endpoint_contract", {}).get("constraints", [])
    ]


def _snippet(value: str, limit: int = 240) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _near_duplicate_pairs(
    examples: list[dict],
    values: list[str],
    threshold: float,
    max_pairs: int,
) -> list[tuple[str, str, float]]:
    normalized = [_normalize_text(value) for value in values]
    token_sets = [set(value.split()) for value in normalized]
    pairs = []
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            overlap = _token_overlap(token_sets[left_index], token_sets[right_index])
            if overlap < threshold - 0.08:
                continue
            score = _sequence_similarity(normalized[left_index], normalized[right_index])
            if score >= threshold and values[left_index] != values[right_index]:
                pairs.append((examples[left_index]["id"], examples[right_index]["id"], score))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _sequence_similarity(left: str, right: str) -> float:
    return max(
        SequenceMatcher(None, left, right).ratio(),
        SequenceMatcher(None, right, left).ratio(),
    )


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
