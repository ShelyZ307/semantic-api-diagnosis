"""Baseline rule-based validator placeholder."""


def validate_example(example: dict) -> dict:
    target = example.get("target", {})
    request = example.get("request", {})
    has_body = isinstance(request.get("body"), dict)
    has_endpoint = bool(example.get("endpoint_family"))
    is_valid = has_body and has_endpoint and target.get("validity") == "valid"
    return {
        "id": example.get("id"),
        "prediction": "valid" if is_valid else "invalid",
        "passed_rules": has_body and has_endpoint,
    }
