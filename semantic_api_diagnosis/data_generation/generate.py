"""Pilot synthetic data generator."""

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from random import Random

from semantic_api_diagnosis.contracts.endpoint_contract import CONTRACTS, get_contract
from semantic_api_diagnosis.contracts.policies import add_counterfactual_policy, apply_policy_to_valid_body
from semantic_api_diagnosis.dataset_plan import SUPPORTED_FAMILY_MODES, SUPPORTED_SPLITS, resolve_families
from semantic_api_diagnosis.error_injectors.injectors import inject_error
from semantic_api_diagnosis.hidden_validators.validator import assert_quality, target_from_findings, validate_example
from semantic_api_diagnosis.serialization.input_text import serialize_model_input
from semantic_api_diagnosis.serialization.jsonl import write_jsonl

ENDPOINT_FAMILIES = tuple(CONTRACTS.keys())
TARGET_LABELS = [
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
    "wrong_type",
    "invalid_value_range",
    "missing_required_field",
    "wrong_http_method",
    "missing_authentication",
    "malformed_url",
    "unexpected_or_malformed_body_structure",
]

LABEL_FAMILIES = {
    label: list(ENDPOINT_FAMILIES)
    for label in TARGET_LABELS
}

MULTI_ERROR_PAIRS = [
    ["semantic_cross_field_violation", "missing_authentication"],
    ["semantic_domain_constraint_violation", "wrong_http_method"],
    ["semantic_state_violation", "malformed_url"],
    ["semantic_state_violation", "missing_required_field"],
    ["invalid_value_range", "malformed_url"],
    ["missing_required_field", "wrong_http_method"],
    ["semantic_cross_field_violation", "wrong_type"],
    ["semantic_domain_constraint_violation", "missing_required_field"],
    ["wrong_type", "malformed_url"],
    ["invalid_value_range", "missing_authentication"],
]


def generate_examples(
    count: int = 10,
    seed: int = 42,
    split: str = "pilot",
    family_mode: str = "implemented",
) -> list[dict]:
    families = resolve_families(split, family_mode)
    rng = Random(seed)
    examples = []
    cases = _planned_cases(count, families)
    for index, case in enumerate(cases):
        labels = case["labels"]
        family = case["family"]
        complexity = case["complexity"]
        example = _build_valid_example(index + 1, family, rng, split)
        for label in labels:
            example = inject_error(example, label, rng)
        _finalize_example(example, complexity)
        examples.append(example)

    assert_quality(examples)
    return examples


def _planned_cases(count: int, families: tuple[str, ...] = ENDPOINT_FAMILIES) -> list[dict]:
    valid_count = round(count * 0.28)
    invalid_count = count - valid_count
    multi_count = min(round(invalid_count * 0.28), invalid_count // 2)
    single_count = invalid_count - multi_count

    invalid_cases = []
    family_counts = Counter()
    label_family_counts = Counter()
    for index in range(single_count):
        label = TARGET_LABELS[index % len(TARGET_LABELS)]
        family = _family_for_labels([label], families, family_counts, label_family_counts)
        invalid_cases.append({"complexity": "single_error", "labels": [label], "family": family})
        family_counts[family] += 1
        _record_label_family_counts([label], family, label_family_counts)
    for index in range(multi_count):
        labels = MULTI_ERROR_PAIRS[index % len(MULTI_ERROR_PAIRS)]
        family = _family_for_labels(labels, families, family_counts, label_family_counts)
        invalid_cases.append({"complexity": "multi_error", "labels": labels, "family": family})
        family_counts[family] += 1
        _record_label_family_counts(labels, family, label_family_counts)

    cases = []
    invalid_index = 0
    valid_index = 0
    for index in range(count):
        remaining_slots = count - index
        remaining_valid = valid_count - valid_index
        should_add_valid = remaining_valid > 0 and (
            index % 4 == 0 or remaining_valid == remaining_slots
        )
        if should_add_valid:
            family = _least_used_family(families, family_counts)
            cases.append(
                {
                    "complexity": "valid",
                    "labels": [],
                    "family": family,
                }
            )
            family_counts[family] += 1
            valid_index += 1
        else:
            cases.append(invalid_cases[invalid_index])
            invalid_index += 1
    return cases


def _family_for_labels(
    labels: list[str],
    families: tuple[str, ...],
    family_counts: Counter,
    label_family_counts: Counter,
) -> str:
    compatible = set(families)
    for label in labels:
        compatible &= set(LABEL_FAMILIES[label])
    if "missing_required_field" in labels and "semantic_domain_constraint_violation" in labels:
        compatible.discard("refunds_orders")
    compatible_families = tuple(family for family in families if family in compatible)
    if not compatible_families:
        raise ValueError(f"No implemented endpoint family can support labels: {', '.join(labels)}")
    primary_label = labels[0]
    return min(
        compatible_families,
        key=lambda family: (
            label_family_counts[(primary_label, family)],
            family_counts[family],
            families.index(family),
        ),
    )


def _record_label_family_counts(labels: list[str], family: str, label_family_counts: Counter) -> None:
    for label in labels:
        label_family_counts[(label, family)] += 1


def _least_used_family(families: tuple[str, ...], family_counts: Counter) -> str:
    return min(families, key=lambda family: (family_counts[family], families.index(family)))


def _build_valid_example(index: int, family: str, rng: Random, split: str) -> dict:
    contract = get_contract(family)
    body = _valid_body(family, rng)
    contract_dict = add_counterfactual_policy(_varied_contract_dict(contract, rng), family, rng)
    apply_policy_to_valid_body(body, contract_dict["policy"], rng)
    request = {
        "method": contract.method,
        "url": _render_url(contract.url_template, body),
        "authentication": "provided" if contract.auth_required else "not_required",
        "query_params": {},
        "body": body,
    }
    return {
        "id": f"ex_{index:06d}",
        "split": split,
        "domain": "semantic_api_request_diagnosis",
        "endpoint_family": family,
        "endpoint_contract": contract_dict,
        "request": request,
        "serialized_input": serialize_model_input(contract_dict, request),
        "target": {
            "validity": "valid",
            "error_labels": [],
            "severity_bucket": "none",
        },
        "generation_metadata": {
            "base_request_was_valid": True,
            "target_error_labels": [],
            "injected_errors": [],
            "num_errors": 0,
            "error_complexity": "valid",
            "noise_level": "low",
        },
    }


def _valid_body(family: str, rng: Random) -> dict:
    if family == "refunds_orders":
        original = rng.choice([49.99, 80.0, 120.5, 200.0])
        refund = round(rng.uniform(5.0, original), 2)
        return {
            "order_id": f"ord_{rng.randint(1000, 9999)}",
            "refund_amount": refund,
            "original_payment": original,
            "order_status": "paid",
            "reason": rng.choice(["duplicate charge", "item damaged", "customer returned item"]),
        }
    if family == "booking_reservation":
        start = date(2026, 6, 1) + timedelta(days=rng.randint(0, 60))
        end = start + timedelta(days=rng.randint(1, 7))
        capacity = rng.choice([2, 4, 6])
        return {
            "reservation_id": f"res_{rng.randint(1000, 9999)}",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "number_of_guests": rng.randint(1, capacity),
            "room_capacity": capacity,
            "reservation_status": rng.choice(["pending", "confirmed"]),
        }
    if family == "user_permissions":
        return {
            "user_id": f"user_{rng.randint(1000, 9999)}",
            "requested_by": f"user_{rng.randint(1000, 9999)}",
            "approved_by": f"manager_{rng.randint(100, 999)}",
            "current_role": "editor",
            "target_role": rng.choice(["manager", "admin"]),
            "approver_role": rng.choice(["manager", "admin"]),
            "request_status": "pending",
        }
    if family == "inventory_product_search":
        product_id = f"prod_{rng.randint(1000, 9999)}"
        region = rng.choice(["us-east", "us-west", "eu-west", "ap-south"])
        available = rng.choice([8, 12, 25, 40])
        return {
            "product_id": product_id,
            "body_product_id": product_id,
            "requested_quantity": rng.randint(1, available),
            "available_stock": available,
            "warehouse_region": region,
            "customer_region": region,
            "reservation_status": rng.choice(["available", "pending"]),
            "allow_backorder": rng.choice([False, True]),
        }
    if family == "payments_invoices":
        invoice_id = f"inv_{rng.randint(1000, 9999)}"
        invoice_total = rng.choice([100.0, 250.0, 500.0, 1200.0])
        amount_paid = round(rng.uniform(0.0, invoice_total * 0.5), 2)
        payment_amount = round(rng.uniform(5.0, invoice_total - amount_paid), 2)
        currency = rng.choice(["USD", "EUR", "GBP"])
        return {
            "invoice_id": invoice_id,
            "body_invoice_id": invoice_id,
            "payment_amount": payment_amount,
            "invoice_total": invoice_total,
            "amount_already_paid": amount_paid,
            "invoice_status": rng.choice(["open", "partially_paid"]),
            "currency": currency,
            "expected_currency": currency,
            "payment_method": rng.choice(["card", "bank_transfer", "wallet"]),
        }
    if family == "healthcare_appointments":
        patient_id = f"pat_{rng.randint(1000, 9999)}"
        start = date(2026, 6, 1) + timedelta(days=rng.randint(0, 60))
        return {
            "patient_id": patient_id,
            "body_patient_id": patient_id,
            "provider_id": f"prov_{rng.randint(100, 999)}",
            "appointment_type": rng.choice(["primary_care", "physical_therapy", "covered_specialist"]),
            "requested_slot_start": f"{start.isoformat()}T10:00",
            "requested_slot_end": f"{start.isoformat()}T10:30",
            "provider_available_from": f"{start.isoformat()}T09:00",
            "provider_available_until": f"{start.isoformat()}T17:00",
            "patient_insurance_status": "active",
            "appointment_status": rng.choice(["draft", "requested"]),
            "requires_referral": False,
            "referral_id": "",
        }
    if family == "course_registration":
        student_id = f"stu_{rng.randint(1000, 9999)}"
        completed = ["MATH101", "ENG100", "CS100"]
        required = rng.sample(completed, k=2)
        credits = rng.choice([3, 4])
        current = rng.choice([6, 9, 12])
        return {
            "student_id": student_id,
            "body_student_id": student_id,
            "course_id": f"course_{rng.randint(100, 999)}",
            "section_id": f"sec_{rng.randint(10, 99)}",
            "completed_prerequisites": completed,
            "required_prerequisites": required,
            "current_credits": current,
            "requested_course_credits": credits,
            "max_allowed_credits": current + credits + 3,
            "enrollment_status": rng.choice(["eligible", "waitlisted"]),
            "registration_window_open": True,
        }
    if family == "ticket_support_workflow":
        ticket_id = f"tkt_{rng.randint(1000, 9999)}"
        agent_id = f"agent_{rng.randint(100, 999)}"
        return {
            "ticket_id": ticket_id,
            "body_ticket_id": ticket_id,
            "current_status": rng.choice(["new", "triaged"]),
            "target_status": rng.choice(["in_progress", "resolved"]),
            "assigned_agent_id": agent_id,
            "acting_user_id": agent_id,
            "customer_visible": rng.choice([True, False]),
            "resolution_code": "fixed",
            "priority": rng.choice(["low", "normal"]),
            "escalation_allowed": True,
        }
    if family == "shipping_returns":
        shipment_id = f"ship_{rng.randint(1000, 9999)}"
        country = rng.choice(["US", "CA", "GB", "DE"])
        window = rng.choice([14, 30, 45])
        return {
            "shipment_id": shipment_id,
            "body_shipment_id": shipment_id,
            "delivery_status": "delivered",
            "days_since_delivery": rng.randint(1, window),
            "return_window_days": window,
            "item_condition": rng.choice(["new", "opened"]),
            "return_reason": rng.choice(["changed_mind", "wrong_size"]),
            "prepaid_label_requested": rng.choice([True, False]),
            "destination_country": country,
            "return_country": country,
        }
    if family == "subscription_plan_changes":
        subscription_id = f"sub_{rng.randint(1000, 9999)}"
        period_end = date(2026, 8, 1) + timedelta(days=rng.randint(0, 60))
        effective = period_end + timedelta(days=rng.randint(0, 10))
        return {
            "subscription_id": subscription_id,
            "body_subscription_id": subscription_id,
            "current_plan": "basic",
            "target_plan": rng.choice(["pro", "enterprise"]),
            "billing_status": "valid",
            "account_status": "active",
            "requested_effective_date": effective.isoformat(),
            "current_billing_period_end": period_end.isoformat(),
            "has_unpaid_invoice": False,
            "downgrade_allowed": True,
        }
    raise ValueError(f"Unsupported endpoint family: {family}")


def _render_url(template: str, body: dict) -> str:
    rendered = template
    for field_name, value in body.items():
        rendered = rendered.replace("{" + field_name + "}", str(value))
    return rendered


def _varied_contract_dict(contract, rng: Random) -> dict:
    contract_dict = contract.to_dataset_dict()
    contract_dict["description"] = rng.choice(contract.description_variants)
    varied_constraints = []
    for constraint in contract.constraints:
        varied = deepcopy(constraint)
        variants = varied.pop("constraint_variants", None)
        if variants:
            varied["constraint_text"] = rng.choice(variants)
        varied_constraints.append(varied)
    contract_dict["constraints"] = varied_constraints
    return contract_dict


def _finalize_example(example: dict, complexity: str) -> None:
    findings = validate_example(example)
    target = target_from_findings(findings)
    example["target"] = target
    metadata = example["generation_metadata"]
    metadata["target_error_labels"] = deepcopy(target["error_labels"])
    metadata["num_errors"] = len(target["error_labels"])
    metadata["error_complexity"] = "valid" if not target["error_labels"] else complexity
    example["serialized_input"] = serialize_model_input(example["endpoint_contract"], example["request"])


def main() -> None:
    parser = ArgumentParser(description="Generate pilot semantic API diagnosis examples.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="pilot")
    parser.add_argument("--family-mode", choices=SUPPORTED_FAMILY_MODES, default="implemented")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    args = parser.parse_args()

    try:
        examples = generate_examples(
            args.count,
            seed=args.seed,
            split=args.split,
            family_mode=args.family_mode,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output:
        write_jsonl(examples, args.output)
        print(f"Wrote {len(examples)} examples to {args.output}")
    else:
        for example in examples:
            print(example)


if __name__ == "__main__":
    main()
