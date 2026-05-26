import subprocess
import sys

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.dataset_plan import implemented_seen_families, implemented_unseen_families


def test_split_argument_is_accepted() -> None:
    examples = generate_examples(count=10, split="validation", family_mode="seen", seed=31)

    assert {example["split"] for example in examples} == {"validation"}


def test_family_mode_argument_is_accepted() -> None:
    examples = generate_examples(count=10, split="pilot", family_mode="implemented", seed=32)

    assert {example["endpoint_family"] for example in examples}


def test_pilot_generation_still_works() -> None:
    examples = generate_examples(count=100, split="pilot", family_mode="implemented", seed=42)

    assert len(examples) == 100


def test_train_generation_uses_only_implemented_seen_families() -> None:
    examples = generate_examples(count=30, split="train", family_mode="seen", seed=33)
    allowed = set(implemented_seen_families())

    assert {example["split"] for example in examples} == {"train"}
    assert {example["endpoint_family"] for example in examples}.issubset(allowed)
    assert allowed.issubset({example["endpoint_family"] for example in examples})


def test_seen_generation_is_balanced_across_implemented_families() -> None:
    examples = generate_examples(count=500, split="train", family_mode="seen", seed=42)
    allowed = set(implemented_seen_families())
    counts = {
        family: sum(1 for example in examples if example["endpoint_family"] == family)
        for family in allowed
    }

    assert set(counts) == allowed
    assert all(70 <= count <= 130 for count in counts.values())


def test_unseen_family_test_uses_only_implemented_unseen_families() -> None:
    examples = generate_examples(count=50, split="unseen_family_test", family_mode="unseen", seed=34)
    allowed = set(implemented_unseen_families())

    assert {example["split"] for example in examples} == {"unseen_family_test"}
    assert {example["endpoint_family"] for example in examples}.issubset(allowed)
    assert allowed.issubset({example["endpoint_family"] for example in examples})


def test_print_dataset_plan_script_runs_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/print_dataset_plan.py"],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Implemented endpoint families:" in result.stdout
    assert "Recommended Version 1 target sizes:" in result.stdout
    assert "implemented seen families: 5/5" in result.stdout
    assert "implemented unseen families: 5/5" in result.stdout
