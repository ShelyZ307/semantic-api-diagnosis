#!/usr/bin/env python3
"""Export a stratified manual review sample from a generated dataset."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.quality_gate import export_manual_review_sample


def main() -> None:
    parser = ArgumentParser(description="Export a stratified manual review sample.")
    parser.add_argument("path")
    parser.add_argument("--examples-per-family", type=int, default=10)
    parser.add_argument("--output", default="data/generated/manual_review_50.jsonl")
    args = parser.parse_args()

    selected = export_manual_review_sample(args.path, args.output, args.examples_per_family)
    print(f"Wrote {len(selected)} examples to {args.output}")
    print(f"Wrote markdown review file to {Path(args.output).with_suffix('.md')}")


if __name__ == "__main__":
    main()
