from random import Random

from semantic_api_diagnosis.data_generation.generate import _build_valid_example, _finalize_example, generate_examples
from semantic_api_diagnosis.dataset_plan import (
    PLANNED_SEEN_ENDPOINT_FAMILIES,
    PLANNED_UNSEEN_ENDPOINT_FAMILIES,
    implemented_seen_families,
    implemented_unseen_families,
)
from semantic_api_diagnosis.error_injectors.injectors import inject_error


SEMANTIC_LABELS = [
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
]


def test_each_unseen_family_can_generate_valid_and_invalid_examples() -> None:
    examples = generate_examples(count=500, split="unseen_family_test", family_mode="unseen", seed=42)
    valid_families = {
        example["endpoint_family"]
        for example in examples
        if example["target"]["validity"] == "valid"
    }
    invalid_families = {
        example["endpoint_family"]
        for example in examples
        if example["target"]["validity"] == "invalid"
    }
    expected = set(PLANNED_UNSEEN_ENDPOINT_FAMILIES)

    assert expected.issubset(valid_families)
    assert expected.issubset(invalid_families)


def test_each_unseen_family_can_produce_required_semantic_labels() -> None:
    for family in PLANNED_UNSEEN_ENDPOINT_FAMILIES:
        for label in SEMANTIC_LABELS:
            example = _build_valid_example(1, family, Random(42), "unseen_family_test")
            example = inject_error(example, label, Random(42))
            _finalize_example(example, "single_error")

            assert label in example["target"]["error_labels"]


def test_seen_and_unseen_family_sets_are_disjoint() -> None:
    assert set(implemented_seen_families()).isdisjoint(implemented_unseen_families())
    assert set(PLANNED_SEEN_ENDPOINT_FAMILIES).isdisjoint(PLANNED_UNSEEN_ENDPOINT_FAMILIES)


def test_train_seen_generation_does_not_include_unseen_families() -> None:
    examples = generate_examples(count=200, split="train", family_mode="seen", seed=42)
    families = {example["endpoint_family"] for example in examples}

    assert families.issubset(set(PLANNED_SEEN_ENDPOINT_FAMILIES))
    assert families.isdisjoint(set(PLANNED_UNSEEN_ENDPOINT_FAMILIES))

