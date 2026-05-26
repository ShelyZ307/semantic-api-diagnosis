"""EDA summaries for generated pilot datasets."""

from collections import Counter

from semantic_api_diagnosis.labels import SEMANTIC_LABELS, STRUCTURAL_LABELS


def summarize_examples(examples: list[dict]) -> dict:
    validity = Counter(example["target"]["validity"] for example in examples)
    endpoint_families = Counter(example["endpoint_family"] for example in examples)
    error_labels = Counter(
        label
        for example in examples
        for label in example["target"]["error_labels"]
    )
    severity_buckets = Counter(example["target"]["severity_bucket"] for example in examples)
    error_complexity = Counter(
        example["generation_metadata"]["error_complexity"]
        for example in examples
    )
    semantic_count = sum(count for label, count in error_labels.items() if label in SEMANTIC_LABELS)
    structural_count = sum(count for label, count in error_labels.items() if label in STRUCTURAL_LABELS)
    return {
        "total_examples": len(examples),
        "validity_distribution": dict(validity),
        "endpoint_family_distribution": dict(endpoint_families),
        "error_label_counts": dict(error_labels),
        "severity_bucket_distribution": dict(severity_buckets),
        "error_complexity_distribution": dict(error_complexity),
        "semantic_vs_structural_label_counts": {
            "semantic": semantic_count,
            "structural": structural_count,
        },
    }
