"""Controlled error injectors for pilot examples."""

from copy import deepcopy
from random import Random


STRUCTURAL_ERROR_LABELS = [
    "missing_required_field",
    "wrong_type",
    "invalid_value_range",
    "malformed_url",
    "wrong_http_method",
    "missing_authentication",
    "unexpected_or_malformed_body_structure",
]

SEMANTIC_BY_FAMILY = {
    "refunds_orders": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "booking_reservation": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "user_permissions": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "inventory_product_search": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "payments_invoices": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "healthcare_appointments": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "course_registration": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "ticket_support_workflow": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "shipping_returns": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
    "subscription_plan_changes": [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ],
}

FAMILY_ERROR_LABELS = {
    family: STRUCTURAL_ERROR_LABELS + semantic_labels
    for family, semantic_labels in SEMANTIC_BY_FAMILY.items()
}


def available_error_labels(endpoint_family: str) -> list[str]:
    return FAMILY_ERROR_LABELS[endpoint_family]


def inject_error(example: dict, label: str, rng: Random | None = None) -> dict:
    rng = rng or Random()
    mutated = deepcopy(example)
    family = mutated["endpoint_family"]
    request = mutated["request"]
    body = request.get("body")

    if label == "missing_required_field" and isinstance(body, dict):
        field = {
            "refunds_orders": "original_payment",
            "booking_reservation": "end_date",
            "user_permissions": "approved_by",
            "inventory_product_search": "body_product_id",
            "payments_invoices": "body_invoice_id",
            "healthcare_appointments": "body_patient_id",
            "course_registration": "body_student_id",
            "ticket_support_workflow": "body_ticket_id",
            "shipping_returns": "body_shipment_id",
            "subscription_plan_changes": "body_subscription_id",
        }[family]
        body.pop(field, None)
    elif label == "wrong_type" and isinstance(body, dict):
        field = {
            "refunds_orders": "refund_amount",
            "booking_reservation": "number_of_guests",
            "user_permissions": "approver_role",
            "inventory_product_search": "requested_quantity",
            "payments_invoices": "payment_amount",
            "healthcare_appointments": "requires_referral",
            "course_registration": "current_credits",
            "ticket_support_workflow": "customer_visible",
            "shipping_returns": "days_since_delivery",
            "subscription_plan_changes": "has_unpaid_invoice",
        }[family]
        body[field] = "not_the_expected_type" if family != "user_permissions" else 7
    elif label == "invalid_value_range" and isinstance(body, dict):
        if family == "refunds_orders":
            body["refund_amount"] = -10.0
        elif family == "booking_reservation":
            body["number_of_guests"] = 0
        elif family == "inventory_product_search":
            body["requested_quantity"] = 0
        elif family == "payments_invoices":
            body["payment_amount"] = -5.0
        elif family == "healthcare_appointments":
            body["appointment_type"] = "unsupported_visit_type"
        elif family == "course_registration":
            body["requested_course_credits"] = 0
        elif family == "ticket_support_workflow":
            body["priority"] = "urgent"
        elif family == "shipping_returns":
            body["days_since_delivery"] = -1
        elif family == "subscription_plan_changes":
            body["target_plan"] = "legacy"
        else:
            body["target_role"] = "superuser"
    elif label == "malformed_url":
        request["url"] = request["url"].replace("/", "_", 1)
    elif label == "wrong_http_method":
        request["method"] = "GET" if request["method"] != "GET" else "DELETE"
    elif label == "missing_authentication":
        request["authentication"] = "missing"
    elif label == "unexpected_or_malformed_body_structure":
        _inject_malformed_body(request, mutated, rng)
    elif family == "refunds_orders":
        _inject_refund_semantic_error(body, label)
    elif family == "booking_reservation":
        _inject_booking_semantic_error(body, label)
    elif family == "user_permissions":
        _inject_permission_semantic_error(body, label, rng)
    elif family == "inventory_product_search":
        _inject_inventory_semantic_error(body, label, rng)
    elif family == "payments_invoices":
        _inject_payment_semantic_error(body, label, rng)
    elif family == "healthcare_appointments":
        _inject_healthcare_semantic_error(body, label, rng)
    elif family == "course_registration":
        _inject_course_semantic_error(body, label, rng)
    elif family == "ticket_support_workflow":
        _inject_ticket_semantic_error(body, label, rng)
    elif family == "shipping_returns":
        _inject_shipping_semantic_error(body, label, rng)
    elif family == "subscription_plan_changes":
        _inject_subscription_semantic_error(body, label, rng)

    metadata = mutated["generation_metadata"]
    metadata["injected_errors"].append(label)
    metadata["target_error_labels"].append(label)
    return mutated


def _inject_malformed_body(request: dict, example: dict, rng: Random) -> None:
    family = example["endpoint_family"]
    example_id = example["id"]
    suffix = example_id.removeprefix("ex_")
    variants = [
        ("array_body", [{"unexpected": f"array_body_{family}_{suffix}"}]),
        ("string_body", f"not_a_json_object_{family}_{suffix}"),
        ("nested_payload", {"payload": {"nested": f"unexpected_{family}_{suffix}"}}),
        ("items_array", {"items": [{"unexpected": True, "family": family, "id": suffix}]}),
        ("data_wrapper", {"data": f"wrong_top_level_wrapper_{family}_{suffix}"}),
    ]
    variant_name, body_value = variants[rng.randrange(len(variants))]
    if variant_name == "missing":
        request.pop("body", None)
    else:
        request["body"] = body_value


def _inject_refund_semantic_error(body: dict, label: str) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_domain_constraint_violation":
        body["refund_amount"] = body["original_payment"] + 25.0
    elif label == "semantic_cross_field_violation":
        body["order_id"] = f"{body['order_id']}_mismatch"
    elif label == "semantic_state_violation":
        body["order_status"] = "pending"


def _inject_booking_semantic_error(body: dict, label: str) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        body["start_date"], body["end_date"] = body["end_date"], body["start_date"]
    elif label == "semantic_domain_constraint_violation":
        body["number_of_guests"] = body["room_capacity"] + 2
    elif label == "semantic_state_violation":
        body["reservation_status"] = "cancelled"


def _inject_permission_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["approved_by"] = body["requested_by"]
        else:
            body["target_role"] = body["current_role"]
    elif label == "semantic_domain_constraint_violation":
        body["current_role"] = "viewer"
        body["target_role"] = "admin"
        body["approver_role"] = rng.choice(["viewer", "editor"])
    elif label == "semantic_state_violation":
        body["request_status"] = "approved"


def _inject_inventory_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_product_id"] = f"{body['product_id']}_mismatch"
        else:
            body["customer_region"] = "eu-west" if body["warehouse_region"] != "eu-west" else "us-east"
    elif label == "semantic_domain_constraint_violation":
        body["requested_quantity"] = body["available_stock"] + 3
        body["allow_backorder"] = False
    elif label == "semantic_state_violation":
        body["reservation_status"] = rng.choice(["cancelled", "unavailable", "locked"])


def _inject_payment_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_invoice_id"] = f"{body['invoice_id']}_mismatch"
        else:
            body["currency"] = "EUR" if body["expected_currency"] != "EUR" else "USD"
    elif label == "semantic_domain_constraint_violation":
        body["payment_amount"] = round(body["invoice_total"] - body["amount_already_paid"] + 10.0, 2)
    elif label == "semantic_state_violation":
        body["invoice_status"] = rng.choice(["cancelled", "closed", "refunded"])


def _inject_healthcare_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_patient_id"] = f"{body['patient_id']}_mismatch"
        else:
            body["requested_slot_start"], body["requested_slot_end"] = (
                body["requested_slot_end"],
                body["requested_slot_start"],
            )
    elif label == "semantic_domain_constraint_violation":
        if rng.random() < 0.5:
            body["requested_slot_start"] = body["provider_available_until"]
            body["requested_slot_end"] = "2026-12-31T23:30"
        else:
            body["requires_referral"] = True
            body.pop("referral_id", None)
    elif label == "semantic_state_violation":
        body["appointment_status"] = rng.choice(["booked", "cancelled", "completed"])


def _inject_course_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        body["body_student_id"] = f"{body['student_id']}_mismatch"
    elif label == "semantic_domain_constraint_violation":
        body["required_prerequisites"] = list(body["completed_prerequisites"]) + ["BIO101"]
    elif label == "semantic_state_violation":
        if rng.random() < 0.5:
            body["registration_window_open"] = False
        else:
            body["enrollment_status"] = rng.choice(["suspended", "graduated", "blocked"])


def _inject_ticket_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_ticket_id"] = f"{body['ticket_id']}_mismatch"
        else:
            body["target_status"] = body["current_status"]
    elif label == "semantic_domain_constraint_violation":
        choice = rng.randrange(3)
        if choice == 0:
            body["target_status"] = "resolved"
            body.pop("resolution_code", None)
        elif choice == 1:
            body["target_status"] = rng.choice(["in_progress", "resolved"])
            body["acting_user_id"] = f"agent_{rng.randint(1000, 9999)}"
        else:
            body["priority"] = "high"
            body["target_status"] = "escalated"
            body["escalation_allowed"] = False
    elif label == "semantic_state_violation":
        body["current_status"] = "closed"
        body["target_status"] = "in_progress"


def _inject_shipping_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_shipment_id"] = f"{body['shipment_id']}_mismatch"
        else:
            body["prepaid_label_requested"] = True
            body["return_country"] = "CA" if body["destination_country"] != "CA" else "US"
    elif label == "semantic_domain_constraint_violation":
        if rng.random() < 0.5:
            body["days_since_delivery"] = body["return_window_days"] + 5
        else:
            body["item_condition"] = "damaged"
            body["return_reason"] = "changed_mind"
    elif label == "semantic_state_violation":
        body["delivery_status"] = rng.choice(["in_transit", "lost", "returned", "cancelled"])


def _inject_subscription_semantic_error(body: dict, label: str, rng: Random) -> None:
    if not isinstance(body, dict):
        return
    if label == "semantic_cross_field_violation":
        if rng.random() < 0.5:
            body["body_subscription_id"] = f"{body['subscription_id']}_mismatch"
        else:
            body["target_plan"] = body["current_plan"]
    elif label == "semantic_domain_constraint_violation":
        if rng.random() < 0.5:
            body["current_plan"] = "enterprise"
            body["target_plan"] = "basic"
            body["requested_effective_date"] = "2026-07-01"
            body["current_billing_period_end"] = "2026-08-01"
            body["downgrade_allowed"] = False
        else:
            body["has_unpaid_invoice"] = True
    elif label == "semantic_state_violation":
        if rng.random() < 0.5:
            body["account_status"] = rng.choice(["paused", "cancelled"])
        else:
            body["current_plan"] = "basic"
            body["target_plan"] = "enterprise"
            body["billing_status"] = rng.choice(["past_due", "failed"])
