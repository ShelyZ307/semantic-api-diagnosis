#!/usr/bin/env python3
"""Check generated datasets for duplicate and near-duplicate examples."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.quality_gate import duplicate_report


def main() -> None:
    parser = ArgumentParser(description="Check duplicates in a generated JSONL dataset.")
    parser.add_argument("path")
    parser.add_argument("--output", default="data/generated/duplicate_report.txt")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--max-groups", type=int, default=10)
    parser.add_argument("--show-examples", choices=["true", "false"], default="false")
    args = parser.parse_args()

    report = duplicate_report(
        args.path,
        args.output,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_groups=args.max_groups,
        show_examples=args.show_examples == "true",
    )
    print(report, end="")


if __name__ == "__main__":
    main()
