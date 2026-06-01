#!/usr/bin/env python3
"""Generate final Version 1 splits with cross-split near-duplicate prevention."""

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.dataset_plan import implemented_seen_families, implemented_unseen_families
from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, SEMANTIC_LABELS
from semantic_api_diagnosis.quality_gate import _normalize_text, _sequence_similarity, _token_overlap
from semantic_api_diagnosis.serialization.jsonl import write_jsonl


DEFAULT_OUTPUTS = {
    "train": "final_train.jsonl",
    "validation": "final_validation.jsonl",
    "seen_test": "final_seen_test.jsonl",
    "unseen_family_test": "final_unseen_family_test.jsonl",
}


def generate_final_splits(
    train_count: int,
    validation_count: int,
    seen_test_count: int,
    unseen_test_count: int,
    output_dir: str | Path,
    seed: int = 101,
    near_duplicate_threshold: float = 0.92,
    max_attempt_batches: int = 40,
) -> dict[str, list[dict]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_index = _SimilarityIndex()
    splits = {
        "train": _generate_filtered_split(
            split="train",
            family_mode="seen",
            target_count=train_count,
            seed=seed,
            reference_index=reference_index,
            threshold=near_duplicate_threshold,
            max_attempt_batches=max_attempt_batches,
        ),
    }
    reference_index.add_split("train", splits["train"])

    splits["validation"] = _generate_filtered_split(
        split="validation",
        family_mode="seen",
        target_count=validation_count,
        seed=seed + 1,
        reference_index=reference_index,
        threshold=near_duplicate_threshold,
        max_attempt_batches=max_attempt_batches,
    )
    reference_index.add_split("validation", splits["validation"])

    splits["seen_test"] = _generate_filtered_split(
        split="seen_test",
        family_mode="seen",
        target_count=seen_test_count,
        seed=seed + 2,
        reference_index=reference_index,
        threshold=near_duplicate_threshold,
        max_attempt_batches=max_attempt_batches,
    )
    reference_index.add_split("seen_test", splits["seen_test"])

    splits["unseen_family_test"] = _generate_filtered_split(
        split="unseen_family_test",
        family_mode="unseen",
        target_count=unseen_test_count,
        seed=seed + 3,
        reference_index=reference_index,
        threshold=near_duplicate_threshold,
        max_attempt_batches=max_attempt_batches,
    )

    for split, rows in splits.items():
        _renumber_examples(rows)
        _assert_split_quality(split, rows)
        write_jsonl(rows, output / DEFAULT_OUTPUTS[split])
    return splits


def _generate_filtered_split(
    split: str,
    family_mode: str,
    target_count: int,
    seed: int,
    reference_index: "_SimilarityIndex",
    threshold: float,
    max_attempt_batches: int,
) -> list[dict]:
    families = implemented_unseen_families() if family_mode == "unseen" else implemented_seen_families()
    family_quota = _balanced_quota(target_count, families)
    accepted = []
    accepted_index = _SimilarityIndex()
    attempts = 0
    batch_size = max(target_count, len(families) * 80)

    while len(accepted) < target_count and attempts < max_attempt_batches:
        batch_seed = seed + attempts * 997
        candidates = generate_examples(batch_size, split=split, family_mode=family_mode, seed=batch_seed)
        for candidate in candidates:
            family = candidate["endpoint_family"]
            if family_quota[family] <= 0:
                continue
            if reference_index.has_near_duplicate(candidate, threshold):
                continue
            if accepted_index.has_near_duplicate(candidate, threshold):
                continue
            accepted.append(deepcopy(candidate))
            accepted_index.add_example(split, candidate)
            family_quota[family] -= 1
            if len(accepted) == target_count:
                break
        attempts += 1

    if len(accepted) != target_count:
        remaining = {family: count for family, count in family_quota.items() if count}
        raise RuntimeError(
            f"Could not generate {target_count} examples for {split} without cross-split near-duplicates. "
            f"Generated {len(accepted)} after {attempts} batches. Remaining family quota: {remaining}"
        )
    return accepted


def _balanced_quota(count: int, families: tuple[str, ...]) -> Counter:
    base = count // len(families)
    remainder = count % len(families)
    return Counter(
        {
            family: base + (1 if index < remainder else 0)
            for index, family in enumerate(families)
        }
    )


def _renumber_examples(examples: list[dict]) -> None:
    for index, example in enumerate(examples, start=1):
        example["id"] = f"ex_{index:06d}"


def _assert_split_quality(split: str, examples: list[dict]) -> None:
    labels = {
        label
        for example in examples
        for label in example.get("target", {}).get("error_labels", [])
    }
    semantic = labels & SEMANTIC_LABELS
    expected_families = (
        set(implemented_unseen_families())
        if split == "unseen_family_test"
        else set(implemented_seen_families())
    )
    families = {example["endpoint_family"] for example in examples}
    if not families.issubset(expected_families):
        raise RuntimeError(f"{split} contains unexpected families: {sorted(families - expected_families)}")
    if labels != FIXED_LABEL_TAXONOMY:
        raise RuntimeError(f"{split} does not contain all labels. Found: {sorted(labels)}")
    if semantic != SEMANTIC_LABELS:
        raise RuntimeError(f"{split} does not contain all semantic labels. Found: {sorted(semantic)}")


class _SimilarityIndex:
    def __init__(self) -> None:
        self.items: list[tuple[str, set[str], str]] = []

    def add_split(self, split: str, examples: list[dict]) -> None:
        for example in examples:
            self.add_example(split, example)

    def add_example(self, split: str, example: dict) -> None:
        normalized = _normalize_text(example.get("serialized_input", ""))
        self.items.append((split, set(normalized.split()), normalized))

    def has_near_duplicate(self, example: dict, threshold: float) -> bool:
        normalized = _normalize_text(example.get("serialized_input", ""))
        tokens = set(normalized.split())
        for _split, existing_tokens, existing in self.items:
            if _token_overlap(tokens, existing_tokens) < threshold - 0.08:
                continue
            if _sequence_similarity(normalized, existing) >= threshold:
                return True
        return False


def main() -> None:
    parser = ArgumentParser(description="Generate final Version 1 dataset splits with leakage prevention.")
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--seen-test-count", type=int, default=700)
    parser.add_argument("--unseen-test-count", type=int, default=700)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--max-attempt-batches", type=int, default=40)
    args = parser.parse_args()

    splits = generate_final_splits(
        train_count=args.train_count,
        validation_count=args.validation_count,
        seen_test_count=args.seen_test_count,
        unseen_test_count=args.unseen_test_count,
        output_dir=args.output_dir,
        seed=args.seed,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_attempt_batches=args.max_attempt_batches,
    )
    for split, rows in splits.items():
        print(f"Wrote {len(rows)} examples to {Path(args.output_dir) / DEFAULT_OUTPUTS[split]}")


if __name__ == "__main__":
    main()
