import subprocess
import sys

from semantic_api_diagnosis.dataset_plan import implemented_seen_families, implemented_unseen_families
from semantic_api_diagnosis.quality_gate import cross_split_leakage_report, duplicate_report
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


def test_generate_final_dataset_script_creates_all_splits_with_strict_leakage_pass(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_final_dataset.py",
            "--train-count",
            "100",
            "--validation-count",
            "50",
            "--seen-test-count",
            "70",
            "--unseen-test-count",
            "70",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "101",
        ],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        check=True,
        capture_output=True,
        text=True,
    )

    expected_counts = {
        "final_train.jsonl": 100,
        "final_validation.jsonl": 50,
        "final_seen_test.jsonl": 70,
        "final_unseen_family_test.jsonl": 70,
    }
    seen = set(implemented_seen_families())
    unseen = set(implemented_unseen_families())
    for filename, expected_count in expected_counts.items():
        rows = read_jsonl(tmp_path / filename)
        families = {row["endpoint_family"] for row in rows}
        assert len(rows) == expected_count
        if filename == "final_unseen_family_test.jsonl":
            assert families.issubset(unseen)
        else:
            assert families.issubset(seen)

    report = cross_split_leakage_report(
        [
            tmp_path / "final_train.jsonl",
            tmp_path / "final_validation.jsonl",
            tmp_path / "final_seen_test.jsonl",
            tmp_path / "final_unseen_family_test.jsonl",
        ],
        output_path=tmp_path / "cross_split_leakage_report.txt",
    )

    assert "strict leakage pass: True" in report
    assert "near-duplicate serialized_input pairs across files >= 0.92: 0" in report
    assert "Wrote 100 examples" in result.stdout

    train_duplicate_report = duplicate_report(
        tmp_path / "final_train.jsonl",
        output_path=tmp_path / "duplicate_report_final_train.txt",
    )
    assert "exact duplicate serialized_input count: 0" in train_duplicate_report
    assert "near-duplicate serialized_input pairs >= 0.92: 0" in train_duplicate_report
