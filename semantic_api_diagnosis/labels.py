"""Shared label taxonomy, label definitions, and severity helpers."""

STRUCTURAL_LABELS = {
    "missing_required_field",
    "wrong_type",
    "invalid_value_range",
    "malformed_url",
    "wrong_http_method",
    "missing_authentication",
    "unexpected_or_malformed_body_structure",
}

SEMANTIC_LABELS = {
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
}

FIXED_LABEL_TAXONOMY = STRUCTURAL_LABELS | SEMANTIC_LABELS

LABEL_DEFINITIONS = {
    "missing_required_field": "A field required by the endpoint contract is absent.",
    "wrong_type": "A present field has the wrong primitive or structured type.",
    "invalid_value_range": (
        "A constrained field has an unsupported value, including numeric values outside "
        "an allowed range, enum/category values outside the allowed set, or other "
        "unsupported values for constrained fields."
    ),
    "malformed_url": "The request URL does not match the endpoint path template.",
    "wrong_http_method": "The request uses a method that does not match the endpoint contract.",
    "missing_authentication": "The endpoint requires authentication but the request omits it.",
    "unexpected_or_malformed_body_structure": (
        "The body is missing, non-object, has the wrong top-level shape, or contains unexpected nesting."
    ),
    "semantic_cross_field_violation": "Two or more present fields conflict with a generic relation.",
    "semantic_domain_constraint_violation": "A present request violates a business/domain rule.",
    "semantic_state_violation": "A status or state field blocks the requested operation.",
}

SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def severity_bucket(severities: list[str]) -> str:
    if not severities:
        return "none"
    return max(severities, key=lambda severity: SEVERITY_RANK[severity])
