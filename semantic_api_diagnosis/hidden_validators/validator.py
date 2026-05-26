"""Hidden machine-checkable validation rules for pilot generation."""

from datetime import date, datetime

from semantic_api_diagnosis.contracts.endpoint_contract import get_contract
from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, severity_bucket


FIELD_SEVERITIES = {
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


def hidden_validate(example: dict) -> bool:
    return not validate_example(example)


def validate_example(example: dict) -> list[dict]:
    family = example["endpoint_family"]
    contract = get_contract(family)
    request = example["request"]
    body_is_present = "body" in request
    body = request.get("body")
    findings = []

    if request.get("method") != contract.method:
        findings.append(_finding("wrong_http_method", "method"))
    if not _url_matches(contract.url_template, request.get("url", "")):
        findings.append(_finding("malformed_url", "url"))
    if contract.auth_required and request.get("authentication") != "provided":
        findings.append(_finding("missing_authentication", "authentication"))

    if (
        not body_is_present
        or not isinstance(body, dict)
        or _has_unexpected_nested_structure(body, contract.required_fields, contract.optional_fields)
        or not _has_any_known_field(body, contract.required_fields)
    ):
        findings.append(_finding("unexpected_or_malformed_body_structure", "body"))
        return _dedupe(findings)

    for field_name, type_spec in contract.required_fields.items():
        if field_name not in body:
            findings.append(_finding("missing_required_field", field_name))
        elif not _matches_type(body[field_name], type_spec):
            findings.append(_finding("wrong_type", field_name))

    findings.extend(_range_findings(family, body))
    findings.extend(_semantic_findings(family, body, request))
    return _dedupe(findings)


def labels_for(example: dict) -> list[str]:
    return [finding["label"] for finding in validate_example(example)]


def target_from_findings(findings: list[dict]) -> dict:
    labels = [finding["label"] for finding in findings]
    severities = [finding["severity"] for finding in findings]
    return {
        "validity": "invalid" if labels else "valid",
        "error_labels": labels,
        "severity_bucket": severity_bucket(severities),
    }


def assert_quality(examples: list[dict]) -> None:
    ids = [example["id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated examples must have unique IDs.")

    for example in examples:
        target = example["target"]
        labels = target["error_labels"]
        if any(label not in FIXED_LABEL_TAXONOMY for label in labels):
            raise ValueError(f"Unknown label in {example['id']}: {labels}")
        if target["validity"] == "valid" and (labels or target["severity_bucket"] != "none"):
            raise ValueError(f"Valid example has errors: {example['id']}")
        if target["validity"] == "invalid" and not labels:
            raise ValueError(f"Invalid example has no labels: {example['id']}")
        if example["generation_metadata"]["error_complexity"] == "multi_error" and len(labels) < 2:
            raise ValueError(f"Multi-error example has fewer than two labels: {example['id']}")
        injected = set(example["generation_metadata"]["injected_errors"])
        detected = set(labels)
        if not injected.issubset(detected):
            raise ValueError(f"Validators did not confirm injected errors for {example['id']}")


def _finding(label: str, field: str) -> dict:
    return {"label": label, "field": field, "severity": FIELD_SEVERITIES[label]}


def _dedupe(findings: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for finding in findings:
        if finding["label"] in seen:
            continue
        seen.add(finding["label"])
        deduped.append(finding)
    return deduped


def _url_matches(template: str, url: str) -> bool:
    prefix, suffix = template.split("{", maxsplit=1)
    suffix = suffix.split("}", maxsplit=1)[1]
    middle = url.removeprefix(prefix)
    if suffix:
        middle = middle.removesuffix(suffix)
    return url.startswith(prefix) and url.endswith(suffix) and bool(middle) and "/" not in middle


def _matches_type(value: object, type_spec: str) -> bool:
    if type_spec == "string":
        return isinstance(value, str) and bool(value)
    if type_spec == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_spec == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_spec == "date":
        return isinstance(value, str) and _parse_date(value) is not None
    if type_spec == "datetime":
        return isinstance(value, str) and _parse_datetime(value) is not None
    if type_spec == "list":
        return isinstance(value, list)
    if type_spec == "boolean":
        return isinstance(value, bool)
    if type_spec.startswith("enum:"):
        return isinstance(value, str)
    return False


def _has_any_known_field(body: dict, required_fields: dict[str, str]) -> bool:
    return any(field_name in body for field_name in required_fields)


def _has_unexpected_nested_structure(
    body: dict,
    required_fields: dict[str, str],
    optional_fields: dict[str, str],
) -> bool:
    allowed_fields = required_fields | optional_fields
    for field_name, value in body.items():
        if isinstance(value, dict):
            return True
        if isinstance(value, list) and allowed_fields.get(field_name) != "list":
            return True
    return False


def _range_findings(family: str, body: dict) -> list[dict]:
    findings = []
    if family == "refunds_orders" and _number(body.get("refund_amount")) is not None:
        if body["refund_amount"] <= 0:
            findings.append(_finding("invalid_value_range", "refund_amount"))
    if family == "booking_reservation":
        guests = body.get("number_of_guests")
        capacity = body.get("room_capacity")
        if isinstance(guests, int) and guests <= 0:
            findings.append(_finding("invalid_value_range", "number_of_guests"))
        if isinstance(capacity, int) and capacity <= 0:
            findings.append(_finding("invalid_value_range", "room_capacity"))
    if family == "user_permissions" and isinstance(body.get("target_role"), str):
        if body["target_role"] not in {"viewer", "editor", "manager", "admin"}:
            findings.append(_finding("invalid_value_range", "target_role"))
    if family == "inventory_product_search":
        requested = body.get("requested_quantity")
        available = body.get("available_stock")
        if isinstance(requested, int) and requested <= 0:
            findings.append(_finding("invalid_value_range", "requested_quantity"))
        if isinstance(available, int) and available < 0:
            findings.append(_finding("invalid_value_range", "available_stock"))
    if family == "payments_invoices":
        payment = _number(body.get("payment_amount"))
        if payment is not None and payment <= 0:
            findings.append(_finding("invalid_value_range", "payment_amount"))
    if family == "healthcare_appointments" and isinstance(body.get("appointment_type"), str):
        if body["appointment_type"] not in {"primary_care", "physical_therapy", "covered_specialist"}:
            findings.append(_finding("invalid_value_range", "appointment_type"))
    if family == "course_registration":
        requested_credits = body.get("requested_course_credits")
        max_credits = body.get("max_allowed_credits")
        if isinstance(requested_credits, int) and requested_credits <= 0:
            findings.append(_finding("invalid_value_range", "requested_course_credits"))
        if isinstance(max_credits, int) and max_credits <= 0:
            findings.append(_finding("invalid_value_range", "max_allowed_credits"))
    if family == "ticket_support_workflow":
        if isinstance(body.get("priority"), str) and body["priority"] not in {"low", "normal", "high"}:
            findings.append(_finding("invalid_value_range", "priority"))
    if family == "shipping_returns":
        days = body.get("days_since_delivery")
        window = body.get("return_window_days")
        if isinstance(days, int) and days < 0:
            findings.append(_finding("invalid_value_range", "days_since_delivery"))
        if isinstance(window, int) and window <= 0:
            findings.append(_finding("invalid_value_range", "return_window_days"))
    if family == "subscription_plan_changes":
        if isinstance(body.get("target_plan"), str) and body["target_plan"] not in {"basic", "pro", "enterprise"}:
            findings.append(_finding("invalid_value_range", "target_plan"))
    return findings


def _semantic_findings(family: str, body: dict, request: dict) -> list[dict]:
    if family == "refunds_orders":
        return _refund_semantics(body, request)
    if family == "booking_reservation":
        return _booking_semantics(body)
    if family == "user_permissions":
        return _permission_semantics(body)
    if family == "inventory_product_search":
        return _inventory_semantics(body, request)
    if family == "payments_invoices":
        return _payment_semantics(body, request)
    if family == "healthcare_appointments":
        return _healthcare_semantics(body, request)
    if family == "course_registration":
        return _course_semantics(body, request)
    if family == "ticket_support_workflow":
        return _ticket_semantics(body, request)
    if family == "shipping_returns":
        return _shipping_semantics(body, request)
    if family == "subscription_plan_changes":
        return _subscription_semantics(body, request)
    return []


def _refund_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_order_id = _path_value("/orders/", "/refunds", request.get("url", ""))
    if url_order_id and _has_fields(body, ["order_id"]) and body["order_id"] != url_order_id:
        findings.append(_finding("semantic_cross_field_violation", "order_id"))
    refund = _number(body.get("refund_amount"))
    original = _number(body.get("original_payment"))
    if refund is not None and original is not None and refund > original:
        findings.append(_finding("semantic_domain_constraint_violation", "refund_amount"))
    if _has_fields(body, ["order_status"]) and body["order_status"] != "paid":
        findings.append(_finding("semantic_state_violation", "order_status"))
    return findings


def _booking_semantics(body: dict) -> list[dict]:
    findings = []
    if _has_fields(body, ["start_date", "end_date"]):
        start = _parse_date(body.get("start_date"))
        end = _parse_date(body.get("end_date"))
        if start and end and start >= end:
            findings.append(_finding("semantic_cross_field_violation", "start_date"))
    guests = body.get("number_of_guests")
    capacity = body.get("room_capacity")
    if isinstance(guests, int) and isinstance(capacity, int) and guests > capacity:
        findings.append(_finding("semantic_domain_constraint_violation", "number_of_guests"))
    if _has_fields(body, ["reservation_status"]) and body["reservation_status"] not in {"pending", "confirmed"}:
        findings.append(_finding("semantic_state_violation", "reservation_status"))
    return findings


def _permission_semantics(body: dict) -> list[dict]:
    findings = []
    if _has_fields(body, ["requested_by", "approved_by"]) and body["requested_by"] == body["approved_by"]:
        findings.append(_finding("semantic_cross_field_violation", "approved_by"))
    if _has_fields(body, ["target_role", "current_role"]) and body["target_role"] == body["current_role"]:
        findings.append(_finding("semantic_cross_field_violation", "target_role"))
    if _role_rank(body.get("target_role")) > _role_rank(body.get("current_role")):
        if isinstance(body.get("approver_role"), str) and body.get("approver_role") not in {"manager", "admin"}:
            findings.append(_finding("semantic_domain_constraint_violation", "approver_role"))
    if _has_fields(body, ["request_status"]) and body["request_status"] != "pending":
        findings.append(_finding("semantic_state_violation", "request_status"))
    return findings


def _inventory_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_product_id = _path_value("/inventory/", "/reservations", request.get("url", ""))
    if url_product_id and _has_fields(body, ["body_product_id"]) and body["body_product_id"] != url_product_id:
        findings.append(_finding("semantic_cross_field_violation", "body_product_id"))
    if _has_fields(body, ["warehouse_region", "customer_region"]) and body["warehouse_region"] != body["customer_region"]:
        findings.append(_finding("semantic_cross_field_violation", "customer_region"))
    requested = body.get("requested_quantity")
    available = body.get("available_stock")
    if isinstance(requested, int) and isinstance(available, int):
        if requested > available and body.get("allow_backorder") is not True:
            findings.append(_finding("semantic_domain_constraint_violation", "requested_quantity"))
    if _has_fields(body, ["reservation_status"]) and body["reservation_status"] not in {"available", "pending"}:
        findings.append(_finding("semantic_state_violation", "reservation_status"))
    return findings


def _payment_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_invoice_id = _path_value("/invoices/", "/payments", request.get("url", ""))
    if url_invoice_id and _has_fields(body, ["body_invoice_id"]) and body["body_invoice_id"] != url_invoice_id:
        findings.append(_finding("semantic_cross_field_violation", "body_invoice_id"))
    if _has_fields(body, ["currency", "expected_currency"]) and body["currency"] != body["expected_currency"]:
        findings.append(_finding("semantic_cross_field_violation", "currency"))
    payment = _number(body.get("payment_amount"))
    invoice_total = _number(body.get("invoice_total"))
    already_paid = _number(body.get("amount_already_paid"))
    if payment is not None and invoice_total is not None and already_paid is not None:
        if payment + already_paid > invoice_total:
            findings.append(_finding("semantic_domain_constraint_violation", "payment_amount"))
    if _has_fields(body, ["invoice_status"]) and body["invoice_status"] not in {"open", "partially_paid"}:
        findings.append(_finding("semantic_state_violation", "invoice_status"))
    if body.get("payment_method") not in {"card", "bank_transfer", "wallet"}:
        findings.append(_finding("invalid_value_range", "payment_method"))
    return findings


def _healthcare_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_patient_id = _path_value("/patients/", "/appointments", request.get("url", ""))
    if url_patient_id and _has_fields(body, ["body_patient_id"]) and body["body_patient_id"] != url_patient_id:
        findings.append(_finding("semantic_cross_field_violation", "body_patient_id"))
    if _has_fields(body, ["requested_slot_start", "requested_slot_end"]):
        start = _parse_datetime(body.get("requested_slot_start"))
        end = _parse_datetime(body.get("requested_slot_end"))
        if start and end and start >= end:
            findings.append(_finding("semantic_cross_field_violation", "requested_slot_start"))
    if _has_fields(
        body,
        [
            "requested_slot_start",
            "requested_slot_end",
            "provider_available_from",
            "provider_available_until",
        ],
    ):
        start = _parse_datetime(body.get("requested_slot_start"))
        end = _parse_datetime(body.get("requested_slot_end"))
        available_from = _parse_datetime(body.get("provider_available_from"))
        available_until = _parse_datetime(body.get("provider_available_until"))
        if start and end and available_from and available_until:
            if start < available_from or end > available_until:
                findings.append(_finding("semantic_domain_constraint_violation", "requested_slot_start"))
    if body.get("requires_referral") is True and not body.get("referral_id"):
        findings.append(_finding("semantic_domain_constraint_violation", "referral_id"))
    if body.get("appointment_type") == "covered_specialist":
        if _has_fields(body, ["patient_insurance_status"]) and body["patient_insurance_status"] != "active":
            findings.append(_finding("semantic_domain_constraint_violation", "patient_insurance_status"))
    if _has_fields(body, ["appointment_status"]) and body["appointment_status"] not in {"draft", "requested"}:
        findings.append(_finding("semantic_state_violation", "appointment_status"))
    return findings


def _course_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_student_id = _path_value("/students/", "/course-registrations", request.get("url", ""))
    if url_student_id and _has_fields(body, ["body_student_id"]) and body["body_student_id"] != url_student_id:
        findings.append(_finding("semantic_cross_field_violation", "body_student_id"))
    if _has_fields(body, ["current_credits", "requested_course_credits", "max_allowed_credits"]):
        current = body.get("current_credits")
        requested = body.get("requested_course_credits")
        maximum = body.get("max_allowed_credits")
        if isinstance(current, int) and isinstance(requested, int) and isinstance(maximum, int):
            if current + requested > maximum:
                findings.append(_finding("semantic_cross_field_violation", "requested_course_credits"))
    if _has_fields(body, ["completed_prerequisites", "required_prerequisites"]):
        completed = body.get("completed_prerequisites")
        required = body.get("required_prerequisites")
        if isinstance(completed, list) and isinstance(required, list):
            if not set(required).issubset(set(completed)):
                findings.append(_finding("semantic_domain_constraint_violation", "required_prerequisites"))
    if _has_fields(body, ["registration_window_open"]) and body["registration_window_open"] is not True:
        findings.append(_finding("semantic_state_violation", "registration_window_open"))
    if _has_fields(body, ["enrollment_status"]) and body["enrollment_status"] not in {"eligible", "waitlisted"}:
        findings.append(_finding("semantic_state_violation", "enrollment_status"))
    return findings


def _ticket_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_ticket_id = _path_value("/tickets/", "/workflow", request.get("url", ""))
    if url_ticket_id and _has_fields(body, ["body_ticket_id"]) and body["body_ticket_id"] != url_ticket_id:
        findings.append(_finding("semantic_cross_field_violation", "body_ticket_id"))
    if _has_fields(body, ["current_status", "target_status"]) and body["current_status"] == body["target_status"]:
        findings.append(_finding("semantic_cross_field_violation", "target_status"))
    if _has_fields(body, ["target_status"]) and body["target_status"] == "resolved" and not body.get("resolution_code"):
        findings.append(_finding("semantic_domain_constraint_violation", "resolution_code"))
    if _has_fields(body, ["assigned_agent_id", "acting_user_id", "target_status"]):
        if body["target_status"] in {"in_progress", "resolved"} and body["acting_user_id"] != body["assigned_agent_id"]:
            findings.append(_finding("semantic_domain_constraint_violation", "acting_user_id"))
    if _has_fields(body, ["priority", "target_status", "escalation_allowed"]):
        if body["priority"] == "high" and body["target_status"] == "escalated" and body["escalation_allowed"] is not True:
            findings.append(_finding("semantic_domain_constraint_violation", "escalation_allowed"))
    if _has_fields(body, ["current_status", "target_status"]):
        if body["current_status"] == "closed" and body["target_status"] == "in_progress":
            findings.append(_finding("semantic_state_violation", "current_status"))
    return findings


def _shipping_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_shipment_id = _path_value("/shipments/", "/returns", request.get("url", ""))
    if url_shipment_id and _has_fields(body, ["body_shipment_id"]) and body["body_shipment_id"] != url_shipment_id:
        findings.append(_finding("semantic_cross_field_violation", "body_shipment_id"))
    if _has_fields(body, ["prepaid_label_requested", "destination_country", "return_country"]):
        if body["prepaid_label_requested"] is True and body["destination_country"] != body["return_country"]:
            findings.append(_finding("semantic_cross_field_violation", "return_country"))
    if _has_fields(body, ["days_since_delivery", "return_window_days"]):
        days = body.get("days_since_delivery")
        window = body.get("return_window_days")
        if isinstance(days, int) and isinstance(window, int) and days > window:
            findings.append(_finding("semantic_domain_constraint_violation", "days_since_delivery"))
    if _has_fields(body, ["item_condition", "return_reason"]):
        if body["item_condition"] in {"damaged", "defective"} and body["return_reason"] not in {"damaged", "defective"}:
            findings.append(_finding("semantic_domain_constraint_violation", "return_reason"))
    if _has_fields(body, ["delivery_status"]) and body["delivery_status"] != "delivered":
        findings.append(_finding("semantic_state_violation", "delivery_status"))
    return findings


def _subscription_semantics(body: dict, request: dict) -> list[dict]:
    findings = []
    url_subscription_id = _path_value("/subscriptions/", "/plan", request.get("url", ""))
    if (
        url_subscription_id
        and _has_fields(body, ["body_subscription_id"])
        and body["body_subscription_id"] != url_subscription_id
    ):
        findings.append(_finding("semantic_cross_field_violation", "body_subscription_id"))
    if _has_fields(body, ["current_plan", "target_plan"]) and body["current_plan"] == body["target_plan"]:
        findings.append(_finding("semantic_cross_field_violation", "target_plan"))
    if _has_fields(body, ["current_plan", "target_plan", "billing_status"]):
        if _plan_rank(body.get("target_plan")) > _plan_rank(body.get("current_plan")):
            if body["billing_status"] != "valid":
                findings.append(_finding("semantic_state_violation", "billing_status"))
    if _has_fields(body, ["requested_effective_date", "current_billing_period_end", "downgrade_allowed"]):
        effective = _parse_date(body.get("requested_effective_date"))
        period_end = _parse_date(body.get("current_billing_period_end"))
        if effective and period_end and effective < period_end and body.get("downgrade_allowed") is not True:
            findings.append(_finding("semantic_domain_constraint_violation", "requested_effective_date"))
    if _has_fields(body, ["has_unpaid_invoice"]) and body["has_unpaid_invoice"] is True:
        findings.append(_finding("semantic_domain_constraint_violation", "has_unpaid_invoice"))
    if _has_fields(body, ["account_status"]) and body["account_status"] != "active":
        findings.append(_finding("semantic_state_violation", "account_status"))
    return findings


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _role_rank(role: object) -> int:
    return {"viewer": 1, "editor": 2, "manager": 3, "admin": 4}.get(role, 0)


def _plan_rank(plan: object) -> int:
    return {"basic": 1, "pro": 2, "enterprise": 3}.get(plan, 0)


def _has_fields(body: dict, fields: list[str]) -> bool:
    return all(field in body for field in fields)


def _path_value(prefix: str, suffix: str, url: str) -> str | None:
    if not isinstance(url, str) or not url.startswith(prefix) or not url.endswith(suffix):
        return None
    value = url.removeprefix(prefix).removesuffix(suffix)
    return value if value and "/" not in value else None
