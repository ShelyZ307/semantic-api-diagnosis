import json

import pytest

from semantic_api_diagnosis.training.data import (
    build_training_records,
    get_error_labels,
    get_serialized_input,
    load_jsonl,
)


def test_load_jsonl_and_build_training_records(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    example = {
        "id": "ex_1",
        "serialized_input": "Endpoint description:\nExample\nRequest:\nMethod: POST",
        "target": {"error_labels": ["missing_required_field"]},
    }
    path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    rows = load_jsonl(path)
    records = build_training_records(path)

    assert rows == [example]
    assert records == [
        {
            "text": example["serialized_input"],
            "labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    ]


def test_extractors_support_top_level_error_labels() -> None:
    example = {"serialized_input": "text", "error_labels": ["wrong_type"]}

    assert get_serialized_input(example) == "text"
    assert get_error_labels(example) == ["wrong_type"]


def test_missing_serialized_input_fails_clearly() -> None:
    with pytest.raises(ValueError, match="serialized_input"):
        get_serialized_input({"target": {"error_labels": []}})


def test_missing_error_labels_fails_clearly() -> None:
    with pytest.raises(ValueError, match="missing error labels"):
        get_error_labels({"serialized_input": "text"})
