from collections import Counter

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.dataset_plan import implemented_seen_families, implemented_unseen_families
from semantic_api_diagnosis.quality_gate import (
    create_shortcut_views,
    cross_split_leakage_report,
    export_manual_review_sample,
    seen_unseen_overlap_report,
)
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


def test_seen_and_unseen_family_sets_are_disjoint_for_leakage_gate() -> None:
    assert set(implemented_seen_families()).isdisjoint(set(implemented_unseen_families()))


def test_seen_unseen_overlap_report_runs_and_writes_report(tmp_path) -> None:
    seen_path = tmp_path / "seen.jsonl"
    unseen_path = tmp_path / "unseen.jsonl"
    report_path = tmp_path / "seen_unseen_overlap_report.txt"
    write_jsonl(generate_examples(50, split="train", family_mode="seen", seed=42), seen_path)
    write_jsonl(generate_examples(50, split="unseen_family_test", family_mode="unseen", seed=42), unseen_path)

    report = seen_unseen_overlap_report(seen_path, unseen_path, output_path=report_path)

    assert report_path.exists()
    assert "Seen vs Unseen Overlap Report" in report
    assert "seen and unseen family sets disjoint: True" in report


def test_cross_split_leakage_checker_runs_and_writes_report(tmp_path) -> None:
    seen_path = tmp_path / "seen.jsonl"
    unseen_path = tmp_path / "unseen.jsonl"
    report_path = tmp_path / "cross_split_leakage_report.txt"
    write_jsonl(generate_examples(50, split="train", family_mode="seen", seed=42), seen_path)
    write_jsonl(generate_examples(50, split="unseen_family_test", family_mode="unseen", seed=42), unseen_path)

    report = cross_split_leakage_report([seen_path, unseen_path], output_path=report_path)

    assert report_path.exists()
    assert "Cross-Split Leakage Report" in report
    assert "exact duplicate serialized_input across files" in report


def test_shortcut_views_for_unseen_are_created(tmp_path) -> None:
    unseen_path = tmp_path / "unseen.jsonl"
    output_dir = tmp_path / "shortcut_views_unseen"
    write_jsonl(generate_examples(50, split="unseen_family_test", family_mode="unseen", seed=42), unseen_path)

    paths = create_shortcut_views(unseen_path, output_dir)

    assert set(paths) == {"request_only", "contract_only", "no_constraints", "family_only"}
    assert all(path.exists() for path in paths.values())


def test_unseen_manual_review_export_creates_50_examples_with_10_per_family(tmp_path) -> None:
    unseen_path = tmp_path / "unseen.jsonl"
    output_path = tmp_path / "manual_review_unseen_50.jsonl"
    write_jsonl(generate_examples(500, split="unseen_family_test", family_mode="unseen", seed=42), unseen_path)

    selected = export_manual_review_sample(unseen_path, output_path, examples_per_family=10)
    counts = Counter(example["endpoint_family"] for example in selected)

    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()
    assert len(read_jsonl(output_path)) == 50
    assert set(counts) == set(implemented_unseen_families())
    assert all(count == 10 for count in counts.values())

