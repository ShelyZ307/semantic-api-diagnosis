from collections import Counter, defaultdict
from random import Random

from semantic_api_diagnosis.contracts.endpoint_contract import CONTRACTS
from semantic_api_diagnosis.data_generation.generate import _build_valid_example, _finalize_example, generate_examples
from semantic_api_diagnosis.error_injectors.injectors import inject_error
from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY


def test_twenty_generated_examples_are_json_objects() -> None:
    examples = generate_examples(count=20, seed=7)

    assert len(examples) == 20
    assert all(isinstance(example, dict) for example in examples)
    assert all(example["id"].startswith("ex_") for example in examples)


def test_all_labels_are_in_fixed_taxonomy() -> None:
    examples = generate_examples(count=40, seed=8)

    for example in examples:
        assert set(example["target"]["error_labels"]).issubset(FIXED_LABEL_TAXONOMY)


def test_valid_examples_have_no_labels_and_no_severity() -> None:
    examples = generate_examples(count=30, seed=9)

    for example in examples:
        if example["target"]["validity"] == "valid":
            assert example["target"]["error_labels"] == []
            assert example["target"]["severity_bucket"] == "none"


def test_invalid_examples_have_labels() -> None:
    examples = generate_examples(count=30, seed=10)

    for example in examples:
        if example["target"]["validity"] == "invalid":
            assert example["target"]["error_labels"]
            assert example["target"]["severity_bucket"] in {"low", "medium", "high"}


def test_serialized_input_does_not_expose_validator_code() -> None:
    examples = generate_examples(count=20, seed=11)
    forbidden_terms = ["hidden_validate", "validate_example", "assert_quality", "FIELD_SEVERITIES"]

    for example in examples:
        serialized = example["serialized_input"]
        assert all(term not in serialized for term in forbidden_terms)


def test_each_endpoint_family_has_valid_and_invalid_examples() -> None:
    examples = generate_examples(count=60, seed=12)
    seen = defaultdict(set)

    for example in examples:
        seen[example["endpoint_family"]].add(example["target"]["validity"])

    assert set(seen) == set(CONTRACTS)
    for validities in seen.values():
        assert {"valid", "invalid"}.issubset(validities)


def test_hundred_example_pilot_contains_all_labels() -> None:
    examples = generate_examples(count=100, seed=42)
    label_counts = Counter(
        label
        for example in examples
        for label in example["target"]["error_labels"]
    )

    assert set(label_counts) == FIXED_LABEL_TAXONOMY
    assert all(count >= 5 for count in label_counts.values())


def test_semantic_cross_field_violation_appears_in_pilot() -> None:
    examples = generate_examples(count=100, seed=42)

    assert any(
        "semantic_cross_field_violation" in example["target"]["error_labels"]
        for example in examples
    )


def test_empty_query_params_serialize_as_plain_none() -> None:
    example = generate_examples(count=1, seed=42)[0]

    assert "Query parameters:\nnone\nBody fields:" in example["serialized_input"]
    assert "- none: none" not in example["serialized_input"]


def test_serialized_inputs_include_varied_contract_language() -> None:
    examples = generate_examples(count=100, seed=42)
    descriptions = {
        example["endpoint_contract"]["description"]
        for example in examples
    }
    constraint_texts = {
        constraint["constraint_text"]
        for example in examples
        for constraint in example["endpoint_contract"]["constraints"]
    }

    assert len(descriptions) >= 6
    assert len(constraint_texts) >= 18


def test_new_seen_families_can_generate_valid_examples() -> None:
    examples = generate_examples(count=150, split="train", family_mode="seen", seed=42)
    valid_families = {
        example["endpoint_family"]
        for example in examples
        if example["target"]["validity"] == "valid"
    }

    assert {"inventory_product_search", "payments_invoices"}.issubset(valid_families)


def test_new_seen_families_can_generate_invalid_examples() -> None:
    examples = generate_examples(count=150, split="train", family_mode="seen", seed=42)
    invalid_families = {
        example["endpoint_family"]
        for example in examples
        if example["target"]["validity"] == "invalid"
    }

    assert {"inventory_product_search", "payments_invoices"}.issubset(invalid_families)


def test_new_seen_families_can_produce_required_semantic_labels() -> None:
    families = ["inventory_product_search", "payments_invoices"]
    labels = [
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ]

    for family in families:
        for label in labels:
            example = _build_valid_example(1, family, Random(42), "train")
            example = inject_error(example, label, Random(42))
            _finalize_example(example, "single_error")

            assert label in example["target"]["error_labels"]
