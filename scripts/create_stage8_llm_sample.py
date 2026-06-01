#!/usr/bin/env python3
"""Create the fixed Stage 8 stratified sample for real LLM baselines."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, SEMANTIC_LABELS
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


SOURCES = {
    "seen_sample": "data/generated/final_seen_test.jsonl",
    "unseen_sample": "data/generated/final_unseen_family_test.jsonl",
    "contract_dependent_unseen_sample": "data/generated/contract_dependent_unseen_test.jsonl",
}
STRUCTURAL_LABELS = sorted(label for label in FIXED_LABEL_TAXONOMY if not label.startswith("semantic_"))
SEMANTIC = sorted(SEMANTIC_LABELS)


def main() -> None:
    parser = ArgumentParser(description="Create the reproducible Stage 8 LLM evaluation sample.")
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--output-dir", default="data/generated/stage8_llm_sample")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "per_source": args.per_source,
        "sources": SOURCES,
        "groups": {},
    }
    combined = []
    for group, source_path in SOURCES.items():
        rows = read_jsonl(source_path)
        sample = _stratified_sample(rows, args.per_source, args.seed + len(combined), group)
        write_jsonl(sample, output_dir / f"{group}.jsonl")
        manifest["groups"][group] = [row["id"] for row in sample]
        combined.extend(sample)
    write_jsonl(combined, output_dir / "combined_sample.jsonl")
    (output_dir / "sampled_ids.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    distribution = _distribution(combined)
    (output_dir / "sample_distribution.json").write_text(
        json.dumps(distribution, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "sample_distribution.md").write_text(
        _markdown_distribution(args.seed, args.per_source, distribution),
        encoding="utf-8",
    )
    print(f"Wrote Stage 8 sample artifacts to {output_dir}")


def _stratified_sample(rows: list[dict], size: int, seed: int, group: str) -> list[dict]:
    rng = Random(seed)
    selected: list[int] = []
    selected_ids = set()
    requirements = [
        lambda row: row["target"]["validity"] == "valid",
        lambda row: len(row["target"]["error_labels"]) > 1,
    ]
    requirements.extend(
        lambda row, label=label: label in row["target"]["error_labels"]
        for label in [*SEMANTIC, *STRUCTURAL_LABELS]
    )
    for requirement in requirements:
        candidates = [index for index, row in enumerate(rows) if index not in selected_ids and requirement(row)]
        if candidates:
            index = rng.choice(candidates)
            selected.append(index)
            selected_ids.add(index)
    remaining = [index for index in range(len(rows)) if index not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    selected = selected[:size]
    selected.sort()
    return [_annotate(rows[index], group) for index in selected]


def _annotate(row: dict, group: str) -> dict:
    annotated = dict(row)
    annotated["sample_group"] = group
    annotated["stage8_id"] = f"{group}:{row['id']}"
    return annotated


def _distribution(rows: list[dict]) -> dict:
    output = {"combined": _group_distribution(rows), "groups": {}}
    for group in SOURCES:
        output["groups"][group] = _group_distribution([row for row in rows if row["sample_group"] == group])
    return output


def _group_distribution(rows: list[dict]) -> dict:
    labels = Counter(label for row in rows for label in row["target"]["error_labels"])
    semantic = Counter({label: count for label, count in labels.items() if label in SEMANTIC})
    families = Counter(row["endpoint_family"] for row in rows)
    validity = Counter(row["target"]["validity"] for row in rows)
    complexity = Counter(row["generation_metadata"]["error_complexity"] for row in rows)
    return {
        "n": len(rows),
        "validity": dict(sorted(validity.items())),
        "labels": dict(sorted(labels.items())),
        "semantic_labels": dict(sorted(semantic.items())),
        "endpoint_families": dict(sorted(families.items())),
        "error_complexity": dict(sorted(complexity.items())),
    }


def _markdown_distribution(seed: int, per_source: int, distribution: dict) -> str:
    lines = [
        "# Stage 8 Fixed LLM Sample Distribution",
        "",
        f"- Seed: `{seed}`.",
        f"- Sampling: greedy stratified coverage, then seeded random fill to `{per_source}` examples per source split.",
        f"- Total sample size: `{distribution['combined']['n']}`.",
        "",
    ]
    for group, stats in distribution["groups"].items():
        lines.extend(_stats_section(group, stats))
    lines.extend(_stats_section("combined", distribution["combined"]))
    return "\n".join(lines)


def _stats_section(name: str, stats: dict) -> list[str]:
    return [
        f"## {name}",
        "",
        f"- Examples: `{stats['n']}`.",
        f"- Validity: `{stats['validity']}`.",
        f"- Semantic labels: `{stats['semantic_labels']}`.",
        f"- Endpoint families: `{stats['endpoint_families']}`.",
        f"- Single vs multi-error: `{stats['error_complexity']}`.",
        f"- Label distribution: `{stats['labels']}`.",
        "",
    ]


if __name__ == "__main__":
    main()
