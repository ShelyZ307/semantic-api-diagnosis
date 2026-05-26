import pytest

from semantic_api_diagnosis.training.labels import ERROR_LABELS, label_to_index, labels_to_multihot, multihot_to_labels


def test_error_label_order_is_stable() -> None:
    assert ERROR_LABELS == [
        "missing_required_field",
        "wrong_type",
        "invalid_value_range",
        "malformed_url",
        "wrong_http_method",
        "missing_authentication",
        "unexpected_or_malformed_body_structure",
        "semantic_cross_field_violation",
        "semantic_domain_constraint_violation",
        "semantic_state_violation",
    ]
    assert len(ERROR_LABELS) == 10
    assert label_to_index()["missing_required_field"] == 0
    assert label_to_index()["semantic_state_violation"] == 9


def test_labels_to_multihot_and_back() -> None:
    vector = labels_to_multihot(["wrong_type", "semantic_state_violation"])

    assert vector == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert multihot_to_labels(vector) == ["wrong_type", "semantic_state_violation"]


def test_unknown_label_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown error label"):
        labels_to_multihot(["new_label"])


def test_multihot_wrong_length_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Expected vector"):
        multihot_to_labels([1, 0])
