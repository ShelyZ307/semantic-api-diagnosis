from collections import Counter

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.quality_gate import (
    create_shortcut_views,
    duplicate_report,
    export_manual_review_sample,
)
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


def test_manual_review_export_creates_50_examples_for_5_families(tmp_path) -> None:
    dataset_path = tmp_path / "seen_500.jsonl"
    output_path = tmp_path / "manual_review_50.jsonl"
    write_jsonl(generate_examples(500, split="train", family_mode="seen", seed=42), dataset_path)

    selected = export_manual_review_sample(dataset_path, output_path, examples_per_family=10)
    counts = Counter(example["endpoint_family"] for example in selected)

    assert output_path.exists()
    assert len(selected) == 50
    assert set(counts.values()) == {10}


def test_manual_review_markdown_is_created(tmp_path) -> None:
    dataset_path = tmp_path / "seen_500.jsonl"
    output_path = tmp_path / "manual_review_50.jsonl"
    write_jsonl(generate_examples(500, split="train", family_mode="seen", seed=42), dataset_path)

    export_manual_review_sample(dataset_path, output_path, examples_per_family=10)

    markdown = output_path.with_suffix(".md")
    assert markdown.exists()
    assert "reviewer_label_correct: yes/no/unsure" in markdown.read_text(encoding="utf-8")


def test_duplicate_checker_runs_and_creates_report(tmp_path) -> None:
    dataset_path = tmp_path / "seen_50.jsonl"
    report_path = tmp_path / "duplicate_report.txt"
    write_jsonl(generate_examples(50, split="train", family_mode="seen", seed=42), dataset_path)

    report = duplicate_report(dataset_path, output_path=report_path)

    assert report_path.exists()
    assert "exact duplicate serialized_input count:" in report
    assert "near-duplicate serialized_input pairs" in report


def test_duplicate_checker_reports_duplicate_request_body_groups(tmp_path) -> None:
    dataset_path = tmp_path / "duplicates.jsonl"
    examples = generate_examples(10, split="train", family_mode="seen", seed=42)
    duplicate = examples[1].copy()
    duplicate["id"] = "duplicate_body_example"
    duplicate["endpoint_family"] = examples[0]["endpoint_family"]
    duplicate["request"] = {
        **duplicate["request"],
        "body": examples[0]["request"]["body"],
    }
    write_jsonl([examples[0], duplicate], dataset_path)

    report = duplicate_report(dataset_path, output_path=tmp_path / "duplicate_report.txt")

    assert "duplicate request-body groups" in report
    assert "duplicate_group_id: body_group_001" in report
    assert "duplicate_body_example" in report


def test_duplicate_checker_reports_near_duplicate_pairs(tmp_path) -> None:
    dataset_path = tmp_path / "near_duplicates.jsonl"
    examples = generate_examples(10, split="train", family_mode="seen", seed=42)
    left = examples[0]
    right = examples[1].copy()
    right["id"] = "near_duplicate_example"
    right["serialized_input"] = left["serialized_input"].replace("provided", "missing", 1)
    write_jsonl([left, right], dataset_path)

    report = duplicate_report(
        dataset_path,
        output_path=tmp_path / "duplicate_report.txt",
        near_duplicate_threshold=0.90,
    )

    assert "near-duplicate pairs:" in report
    assert "similarity_score:" in report
    assert "difference_summary:" in report


def test_shortcut_views_are_created_and_preserve_targets(tmp_path) -> None:
    dataset_path = tmp_path / "seen_50.jsonl"
    output_dir = tmp_path / "shortcut_views"
    source = generate_examples(50, split="train", family_mode="seen", seed=42)
    write_jsonl(source, dataset_path)

    paths = create_shortcut_views(dataset_path, output_dir)

    assert set(paths) == {"request_only", "contract_only", "no_constraints", "family_only"}
    for path in paths.values():
        assert path.exists()
        rows = read_jsonl(path)
        assert [row["target"] for row in rows] == [row["target"] for row in source]


def test_request_only_view_does_not_contain_required_constraints(tmp_path) -> None:
    dataset_path = tmp_path / "seen_10.jsonl"
    output_dir = tmp_path / "shortcut_views"
    write_jsonl(generate_examples(10, split="train", family_mode="seen", seed=42), dataset_path)

    paths = create_shortcut_views(dataset_path, output_dir)
    rows = read_jsonl(paths["request_only"])

    assert all("Required constraints" not in row["serialized_input"] for row in rows)


def test_contract_only_view_does_not_contain_body_fields(tmp_path) -> None:
    dataset_path = tmp_path / "seen_10.jsonl"
    output_dir = tmp_path / "shortcut_views"
    write_jsonl(generate_examples(10, split="train", family_mode="seen", seed=42), dataset_path)

    paths = create_shortcut_views(dataset_path, output_dir)
    rows = read_jsonl(paths["contract_only"])

    assert all("Body fields" not in row["serialized_input"] for row in rows)


def test_no_constraints_view_does_not_contain_required_constraints(tmp_path) -> None:
    dataset_path = tmp_path / "seen_10.jsonl"
    output_dir = tmp_path / "shortcut_views"
    write_jsonl(generate_examples(10, split="train", family_mode="seen", seed=42), dataset_path)

    paths = create_shortcut_views(dataset_path, output_dir)
    rows = read_jsonl(paths["no_constraints"])

    assert all("Required constraints" not in row["serialized_input"] for row in rows)


def test_family_only_view_contains_only_endpoint_family_text(tmp_path) -> None:
    dataset_path = tmp_path / "seen_10.jsonl"
    output_dir = tmp_path / "shortcut_views"
    write_jsonl(generate_examples(10, split="train", family_mode="seen", seed=42), dataset_path)

    paths = create_shortcut_views(dataset_path, output_dir)
    rows = read_jsonl(paths["family_only"])

    assert all(row["serialized_input"].startswith("endpoint_family: ") for row in rows)
    assert all("\n" not in row["serialized_input"] for row in rows)


def test_malformed_body_examples_cover_multiple_endpoint_families() -> None:
    examples = generate_examples(500, split="train", family_mode="seen", seed=42)
    malformed_families = {
        example["endpoint_family"]
        for example in examples
        if "unexpected_or_malformed_body_structure" in example["target"]["error_labels"]
    }

    assert len(malformed_families) >= 3


def test_malformed_body_shapes_vary() -> None:
    examples = generate_examples(500, split="train", family_mode="seen", seed=42)
    malformed_shapes = {
        str(example["request"].get("body", "<missing>"))
        for example in examples
        if "unexpected_or_malformed_body_structure" in example["target"]["error_labels"]
    }

    assert len(malformed_shapes) >= 6


def test_duplicate_checker_no_large_repeated_malformed_body_warning(tmp_path) -> None:
    dataset_path = tmp_path / "seen_500.jsonl"
    write_jsonl(generate_examples(500, split="train", family_mode="seen", seed=42), dataset_path)

    report = duplicate_report(dataset_path, output_path=tmp_path / "duplicate_report.txt")

    assert "repeats family=payments_invoices labels=unexpected_or_malformed_body_structure" not in report
    assert "unexpected_or_malformed_body_structure 25 times" not in report
