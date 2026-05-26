#!/usr/bin/env python3
"""Check generated dataset files for cross-split leakage."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.quality_gate import cross_split_leakage_report


def main() -> None:
    parser = ArgumentParser(description="Check serialized examples for cross-split leakage.")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", default="data/generated/cross_split_leakage_report.txt")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    args = parser.parse_args()

    report = cross_split_leakage_report(
        args.splits,
        output_path=args.output,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(report, end="")


if __name__ == "__main__":
    main()
