"""Canonical training label order and multi-hot conversion helpers."""

ERROR_LABELS = [
    "missing_required_field",
    "wrong_type",
    "invalid_value_range",
    "malformed_url",
    "wrong_http_method",
    "missing_authentication",
    "unexpected_or_malformed_body_structure",
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
]


def label_to_index() -> dict[str, int]:
    return {label: index for index, label in enumerate(ERROR_LABELS)}


def labels_to_multihot(labels: list[str]) -> list[int]:
    indices = label_to_index()
    vector = [0] * len(ERROR_LABELS)
    for label in labels:
        if label not in indices:
            raise ValueError(f"Unknown error label: {label}")
        vector[indices[label]] = 1
    return vector


def multihot_to_labels(vector, threshold: float | list[float] = 0.5) -> list[str]:
    values = list(vector)
    if len(values) != len(ERROR_LABELS):
        raise ValueError(f"Expected vector of length {len(ERROR_LABELS)}, got {len(values)}")
    thresholds = [float(threshold)] * len(values) if isinstance(threshold, (int, float)) else list(threshold)
    if len(thresholds) != len(values):
        raise ValueError(f"Expected threshold vector of length {len(values)}, got {len(thresholds)}")
    return [
        label
        for label, value, cutoff in zip(ERROR_LABELS, values, thresholds)
        if float(value) >= float(cutoff)
    ]
