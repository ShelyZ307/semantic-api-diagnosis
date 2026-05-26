from copy import deepcopy

from semantic_api_diagnosis.audit import audit_dataset_file, build_audit_report
from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.serialization.jsonl import write_jsonl


def test_audit_script_core_runs_on_generated_temporary_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "pilot.jsonl"
    report_path = tmp_path / "audit.txt"
    write_jsonl(generate_examples(count=12, seed=21), dataset_path)

    report, warnings = audit_dataset_file(dataset_path, output_path=report_path)

    assert report_path.exists()
    assert "total examples: 12" in report
    assert "## Basic Counts" in report
    assert warnings == []


def test_audit_detects_basic_counts_correctly() -> None:
    examples = generate_examples(count=6, seed=22)

    report, warnings = build_audit_report(examples)

    assert "total examples: 6" in report
    assert "valid: 2 (33.3%)" in report
    assert "invalid: 4 (66.7%)" in report
    assert warnings == []


def test_audit_warns_if_serialized_input_is_missing() -> None:
    examples = generate_examples(count=3, seed=23)
    broken = deepcopy(examples[0])
    broken["serialized_input"] = ""

    _report, warnings = build_audit_report([broken])

    assert any("serialized_input is missing or empty" in warning for warning in warnings)


def test_audit_warns_if_valid_example_has_labels() -> None:
    example = generate_examples(count=4, seed=24)[0]
    example["target"]["error_labels"] = ["missing_required_field"]

    _report, warnings = build_audit_report([example])

    assert any("valid example has error labels" in warning for warning in warnings)


def test_audit_warns_if_invalid_example_has_no_labels() -> None:
    example = generate_examples(count=4, seed=25)[3]
    example["target"]["validity"] = "invalid"
    example["target"]["error_labels"] = []
    example["target"]["severity_bucket"] = "none"
    example["generation_metadata"]["num_errors"] = 0

    _report, warnings = build_audit_report([example])

    assert any("invalid example has no error labels" in warning for warning in warnings)
