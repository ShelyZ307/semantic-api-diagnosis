from copy import deepcopy
from random import Random

from semantic_api_diagnosis.data_generation.generate import _build_valid_example
from semantic_api_diagnosis.hidden_validators.validator import labels_for


def _example_for_family(family: str) -> dict:
    return _build_valid_example(1, family, Random(42), "train")


def test_missing_body_invoice_id_does_not_trigger_cross_field_violation() -> None:
    example = _example_for_family("payments_invoices")
    example["request"]["body"].pop("body_invoice_id")

    labels = labels_for(example)

    assert "missing_required_field" in labels
    assert "semantic_cross_field_violation" not in labels


def test_missing_refund_order_id_does_not_trigger_cross_field_violation() -> None:
    example = _example_for_family("refunds_orders")
    example["request"]["body"].pop("order_id")

    labels = labels_for(example)

    assert "missing_required_field" in labels
    assert "semantic_cross_field_violation" not in labels


def test_missing_body_product_id_does_not_trigger_cross_field_violation() -> None:
    example = _example_for_family("inventory_product_search")
    example["request"]["body"].pop("body_product_id")

    labels = labels_for(example)

    assert "missing_required_field" in labels
    assert "semantic_cross_field_violation" not in labels


def test_missing_one_date_field_does_not_trigger_cross_field_violation() -> None:
    example = _example_for_family("booking_reservation")
    example["request"]["body"].pop("end_date")

    labels = labels_for(example)

    assert "missing_required_field" in labels
    assert "semantic_cross_field_violation" not in labels


def test_missing_currency_or_expected_currency_does_not_trigger_cross_field_violation() -> None:
    for field in ["currency", "expected_currency"]:
        example = _example_for_family("payments_invoices")
        example["request"]["body"].pop(field)

        labels = labels_for(example)

        assert "missing_required_field" in labels
        assert "semantic_cross_field_violation" not in labels


def test_refunds_missing_reason_does_not_trigger_missing_required_field() -> None:
    example = _example_for_family("refunds_orders")
    example["request"]["body"].pop("reason")

    labels = labels_for(example)

    assert "missing_required_field" not in labels


def test_missing_order_status_does_not_trigger_state_violation() -> None:
    example = _example_for_family("refunds_orders")
    example["request"]["body"].pop("order_status")

    labels = labels_for(example)

    assert "missing_required_field" in labels
    assert "semantic_state_violation" not in labels


def test_cross_field_violation_still_emits_when_both_fields_present_and_conflict() -> None:
    example = _example_for_family("payments_invoices")
    body = deepcopy(example["request"]["body"])
    body["body_invoice_id"] = f"{body['invoice_id']}_mismatch"
    example["request"]["body"] = body

    labels = labels_for(example)

    assert "semantic_cross_field_violation" in labels


def test_enum_like_unsupported_values_can_emit_invalid_value_range() -> None:
    example = _example_for_family("subscription_plan_changes")
    example["request"]["body"]["target_plan"] = "legacy"

    labels = labels_for(example)

    assert "invalid_value_range" in labels
