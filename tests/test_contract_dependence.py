from pathlib import Path

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.evaluation.contract_dependence import analyze_contract_dependence, classify_example
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


def test_contract_dependence_classifies_expected_buckets() -> None:
    valid = _example_with_labels([])
    request_obvious = _example_with_labels(["missing_authentication"])
    contract_dependent = _example_with_labels(["semantic_state_violation"])
    mixed = _example_with_labels(["missing_authentication", "semantic_state_violation"])

    assert classify_example(valid) == "valid"
    assert classify_example(request_obvious) == "request_obvious"
    assert classify_example(contract_dependent) == "contract_dependent"
    assert classify_example(mixed) == "mixed"


def test_contract_dependence_script_outputs_report_and_subsets(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    seen_path = tmp_path / "seen_test.jsonl"
    unseen_path = tmp_path / "unseen_test.jsonl"
    output_path = tmp_path / "contract_dependence_report.md"
    write_jsonl(generate_examples(100, split="train", family_mode="seen", seed=101), train_path)
    write_jsonl(generate_examples(80, split="seen_test", family_mode="seen", seed=103), seen_path)
    write_jsonl(generate_examples(80, split="unseen_family_test", family_mode="unseen", seed=104), unseen_path)

    result = analyze_contract_dependence(train_path, seen_path, unseen_path, output_path)

    assert output_path.exists()
    assert "Contract Dependence Analysis" in output_path.read_text(encoding="utf-8")
    assert (tmp_path / "contract_dependent_seen_test.jsonl").exists()
    assert (tmp_path / "contract_dependent_unseen_test.jsonl").exists()
    assert read_jsonl(tmp_path / "contract_dependent_seen_test.jsonl")
    assert read_jsonl(tmp_path / "contract_dependent_unseen_test.jsonl")
    assert "seen_test" in result["analysis"]


def _example_with_labels(labels: list[str]) -> dict:
    example = generate_examples(1, split="train", family_mode="seen", seed=101)[0]
    example["target"]["error_labels"] = labels
    example["target"]["validity"] = "invalid" if labels else "valid"
    example["target"]["severity_bucket"] = "high" if labels else "none"
    return example
