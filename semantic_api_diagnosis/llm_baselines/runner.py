"""LLM baseline runner with mock and dry-run support."""

import json
import os
import re

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, severity_bucket


LABEL_SEVERITIES = {
    "missing_required_field": "high",
    "wrong_type": "medium",
    "invalid_value_range": "medium",
    "malformed_url": "medium",
    "wrong_http_method": "high",
    "missing_authentication": "high",
    "unexpected_or_malformed_body_structure": "high",
    "semantic_cross_field_violation": "high",
    "semantic_domain_constraint_violation": "high",
    "semantic_state_violation": "high",
}


class LLMRunner:
    def __init__(self, provider: str, model: str, dry_run: bool = False) -> None:
        self.provider = provider
        self.model = model
        self.dry_run = dry_run

    def run(self, prompt: str) -> dict:
        if self.dry_run or self.provider == "manual":
            return {
                "raw_response": "",
                "parsed_prediction": None,
                "parse_error": "dry_run_no_response",
            }
        if self.provider == "mock":
            raw = json.dumps(_mock_prediction(prompt), sort_keys=True)
            return _parse_response(raw)
        if self.provider == "openai":
            raw = self._run_openai(prompt)
            return _parse_response(raw)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _run_openai(self, prompt: str) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for provider=openai")
        raise NotImplementedError(
            "OpenAI provider wiring is intentionally not enabled yet. Use provider=mock or dry-run."
        )


def parse_prediction_response(raw_response: str) -> dict:
    return _parse_response(raw_response)


def _parse_response(raw_response: str) -> dict:
    try:
        parsed = json.loads(raw_response)
        prediction = _normalize_prediction(parsed)
        return {"raw_response": raw_response, "parsed_prediction": prediction, "parse_error": None}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {"raw_response": raw_response, "parsed_prediction": None, "parse_error": str(error)}


def _normalize_prediction(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    validity = parsed.get("validity")
    if validity not in {"valid", "invalid"}:
        raise ValueError("validity must be valid or invalid")
    labels = parsed.get("error_labels")
    if not isinstance(labels, list):
        raise ValueError("error_labels must be a list")
    clean_labels = []
    for label in labels:
        if label not in FIXED_LABEL_TAXONOMY:
            raise ValueError(f"unknown error label: {label}")
        if label not in clean_labels:
            clean_labels.append(label)
    severity = parsed.get("severity_bucket")
    if severity not in {"none", "medium", "high"}:
        raise ValueError("severity_bucket must be none, medium, or high")
    return {"validity": validity, "error_labels": clean_labels, "severity_bucket": severity}


def _mock_prediction(prompt: str) -> dict:
    request_text = prompt.split("Serialized input:")[-1]
    labels = set()
    if "Authentication: missing" in request_text:
        labels.add("missing_authentication")
    if re.search(r"Method: GET\b", request_text):
        labels.add("wrong_http_method")
    url_match = re.search(r"URL: (.+)", request_text)
    if url_match and not url_match.group(1).startswith("/"):
        labels.add("malformed_url")
    if "not_the_expected_type" in request_text:
        labels.add("wrong_type")
    if re.search(r"- body: (\[\]|None|null|not_a_json_object)", request_text):
        labels.add("unexpected_or_malformed_body_structure")
    if any(marker in request_text for marker in ["unsupported_visit_type", "superuser", "legacy", "- priority: urgent"]):
        labels.add("invalid_value_range")
    if re.search(r"- [a-z_]*(amount|quantity|guests|capacity|days|credits|stock|payment)[a-z_]*: -", request_text):
        labels.add("invalid_value_range")
    if "_mismatch" in request_text or _current_target_same(request_text):
        labels.add("semantic_cross_field_violation")
    if any(status in request_text for status in ["cancelled", "closed", "blocked", "lost", "returned", "past_due", "failed"]):
        labels.add("semantic_state_violation")
    ordered = sorted(labels)
    return {
        "validity": "invalid" if ordered else "valid",
        "error_labels": ordered,
        "severity_bucket": severity_bucket([LABEL_SEVERITIES[label] for label in ordered]),
    }


def _current_target_same(text: str) -> bool:
    values = {}
    for line in text.splitlines():
        if not line.startswith("- "):
            continue
        name, sep, value = line[2:].partition(": ")
        if sep:
            values[name] = value
    for name, value in values.items():
        if name.startswith("current_"):
            target = "target_" + name.removeprefix("current_")
            if values.get(target) == value:
                return True
    return False
