"""Transparent visible-text rule baseline.

This module intentionally does not import hidden validators. Rules use only the
serialized input and public example metadata.
"""

from datetime import date, datetime
import ast
import re

from semantic_api_diagnosis.labels import SEMANTIC_LABELS, severity_bucket


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


class VisibleRuleBasedBaseline:
    def fit(self, examples: list[dict]) -> "VisibleRuleBasedBaseline":
        return self

    def predict_one(self, example: dict) -> dict:
        text = example.get("serialized_input", "")
        fields = _parse_body_fields(text)
        labels = set()
        method = _line_value(text, "Method")
        url = _line_value(text, "URL")
        authentication = _line_value(text, "Authentication")
        if authentication == "missing":
            labels.add("missing_authentication")
        if method == "GET":
            labels.add("wrong_http_method")
        if url and not url.startswith("/"):
            labels.add("malformed_url")
        body_value = fields.get("body")
        if body_value is not None or any(wrapper in fields for wrapper in ["payload", "items", "data"]):
            labels.add("unexpected_or_malformed_body_structure")
        if any(value == "not_the_expected_type" for value in fields.values()):
            labels.add("wrong_type")
        if _visible_minimum_violation(text, fields):
            labels.add("invalid_value_range")
        elif any(isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0 for value in fields.values()):
            labels.add("invalid_value_range")
        if _visible_allowed_value_violation(text, fields):
            labels.add("invalid_value_range")
        elif _has_unsupported_enum_marker(fields):
            labels.add("invalid_value_range")
        if _visible_cross_field_violation(fields, url):
            labels.add("semantic_cross_field_violation")
        if _visible_domain_violation(text, fields):
            labels.add("semantic_domain_constraint_violation")
        if _visible_state_violation(text, fields):
            labels.add("semantic_state_violation")
        ordered = sorted(labels)
        severities = [LABEL_SEVERITIES[label] for label in ordered]
        return {
            "error_labels": ordered,
            "validity": "invalid" if ordered else "valid",
            "severity_bucket": severity_bucket(severities),
        }

    def predict(self, examples: list[dict]) -> list[dict]:
        return [self.predict_one(example) for example in examples]


def _parse_body_fields(text: str) -> dict:
    fields = {}
    in_body = False
    for line in text.splitlines():
        if line.strip() == "Body fields:":
            in_body = True
            continue
        if in_body and line and not line.startswith("- "):
            break
        if not in_body or not line.startswith("- "):
            continue
        name, sep, value = line[2:].partition(": ")
        if sep:
            fields[name] = _parse_value(value)
    return fields


def _line_value(text: str, name: str) -> str | None:
    prefix = f"{name}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return None


def _parse_value(value: str) -> object:
    if value in {"True", "False"}:
        return value == "True"
    if value in {"None", "null"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _has_unsupported_enum_marker(fields: dict) -> bool:
    unsupported_values = {"superuser", "unsupported_visit_type", "urgent", "legacy"}
    return any(isinstance(value, str) and value in unsupported_values for value in fields.values())


def _visible_cross_field_violation(fields: dict, url: str | None) -> bool:
    for field, value in fields.items():
        if field.startswith("body_") and field.endswith("_id"):
            plain_field = field.removeprefix("body_")
            if plain_field in fields and fields[plain_field] != value:
                return True
            if url and isinstance(value, str) and _path_identifier(url) and value != _path_identifier(url):
                return True
    for current_field, current_value in fields.items():
        if current_field.startswith("current_"):
            target_field = "target_" + current_field.removeprefix("current_")
            if target_field in fields and fields[target_field] == current_value:
                return True
    if _date_order_bad(fields.get("start_date"), fields.get("end_date")):
        return True
    if _date_order_bad(fields.get("requested_slot_start"), fields.get("requested_slot_end")):
        return True
    if fields.get("currency") and fields.get("expected_currency") and fields["currency"] != fields["expected_currency"]:
        return True
    if fields.get("warehouse_region") and fields.get("customer_region") and fields["warehouse_region"] != fields["customer_region"]:
        return True
    if fields.get("prepaid_label_requested") is True and fields.get("destination_country") != fields.get("return_country"):
        return True
    return False


def _visible_domain_violation(text: str, fields: dict) -> bool:
    if _visible_maximum_violation(text, fields):
        return True
    if _number(fields.get("refund_amount")) and _number(fields.get("original_payment")):
        if fields["refund_amount"] > fields["original_payment"]:
            return True
    if all(_number(fields.get(name)) is not None for name in ["payment_amount", "amount_already_paid", "invoice_total"]):
        if fields["payment_amount"] + fields["amount_already_paid"] > fields["invoice_total"]:
            return True
    if isinstance(fields.get("number_of_guests"), int) and isinstance(fields.get("room_capacity"), int):
        if fields["number_of_guests"] > fields["room_capacity"]:
            return True
    if isinstance(fields.get("requested_quantity"), int) and isinstance(fields.get("available_stock"), int):
        if fields["requested_quantity"] > fields["available_stock"] and fields.get("allow_backorder") is not True:
            return True
    if isinstance(fields.get("days_since_delivery"), int) and isinstance(fields.get("return_window_days"), int):
        if fields["days_since_delivery"] > fields["return_window_days"]:
            return True
    if isinstance(fields.get("completed_prerequisites"), list) and isinstance(fields.get("required_prerequisites"), list):
        if not set(fields["required_prerequisites"]).issubset(set(fields["completed_prerequisites"])):
            return True
    if fields.get("requires_referral") is True and not fields.get("referral_id"):
        return True
    if fields.get("has_unpaid_invoice") is True:
        return True
    return False


def _visible_state_violation(text: str, fields: dict) -> bool:
    blocked_by_field = {
        "order_status": {"pending", "cancelled", "refunded"},
        "reservation_status": {"checked_in", "cancelled"},
        "request_status": {"approved", "rejected"},
        "reservation_status": {"cancelled", "unavailable", "locked", "checked_in"},
        "invoice_status": {"cancelled", "closed", "refunded"},
        "appointment_status": {"booked", "cancelled", "completed"},
        "enrollment_status": {"suspended", "graduated", "blocked"},
        "delivery_status": {"in_transit", "lost", "returned", "cancelled"},
        "account_status": {"paused", "cancelled"},
    }
    dynamic_rules = _allowed_value_rules(text)
    for field in _state_fields():
        if field in dynamic_rules and field in fields:
            if str(fields[field]) not in dynamic_rules[field]:
                return True
    for field, blocked_values in blocked_by_field.items():
        if field in dynamic_rules:
            continue
        if fields.get(field) in blocked_values:
            return True
    if fields.get("registration_window_open") is False:
        return True
    if fields.get("billing_status") in {"past_due", "failed"}:
        return True
    if fields.get("current_status") == "closed" and fields.get("target_status") == "in_progress":
        return True
    return False


def _visible_minimum_violation(text: str, fields: dict) -> bool:
    for field, value in re.findall(r"For this endpoint, ([a-z_]+) must be at least ([0-9.]+)\.", text):
        if _number(fields.get(field)) is not None and fields[field] < float(value):
            return True
    return False


def _visible_maximum_violation(text: str, fields: dict) -> bool:
    for field, value in re.findall(r"For this endpoint, ([a-z_]+) must not exceed ([0-9.]+)\.", text):
        if _number(fields.get(field)) is not None and fields[field] > float(value):
            return True
    return False


def _visible_allowed_value_violation(text: str, fields: dict) -> bool:
    rules = _allowed_value_rules(text)
    return any(
        field not in _state_fields() and field in fields and str(fields[field]) not in allowed
        for field, allowed in rules.items()
    )


def _allowed_value_rules(text: str) -> dict[str, set[str]]:
    return {
        field: {value.strip() for value in values.split(",")}
        for field, values in re.findall(r"For this endpoint, ([a-z_]+) must be one of: ([^.]+)\.", text)
    }


def _state_fields() -> set[str]:
    return {
        "order_status",
        "reservation_status",
        "request_status",
        "invoice_status",
        "appointment_status",
        "enrollment_status",
        "current_status",
        "delivery_status",
        "account_status",
    }


def _path_identifier(url: str) -> str | None:
    parts = [part for part in url.split("/") if part]
    return parts[1] if len(parts) >= 2 else None


def _date_order_bad(left: object, right: object) -> bool:
    left_parsed = _parse_temporal(left)
    right_parsed = _parse_temporal(right)
    return bool(left_parsed and right_parsed and left_parsed >= right_parsed)


def _parse_temporal(value: object):
    if not isinstance(value, str):
        return None
    for parser in (datetime.fromisoformat, date.fromisoformat):
        try:
            return parser(value)
        except ValueError:
            continue
    return None


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
