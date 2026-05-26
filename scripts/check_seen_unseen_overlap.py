#!/usr/bin/env python3
"""Compare seen and unseen diagnostic datasets for overlap and shortcut risk."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.quality_gate import seen_unseen_overlap_report


def main() -> None:
    parser = ArgumentParser(description="Check overlap between seen and unseen generated datasets.")
    parser.add_argument("--seen", required=True)
    parser.add_argument("--unseen", required=True)
    parser.add_argument("--output", default="data/generated/seen_unseen_overlap_report.txt")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    args = parser.parse_args()

    report = seen_unseen_overlap_report(
        args.seen,
        args.unseen,
        output_path=args.output,
        similarity_threshold=args.similarity_threshold,
    )
    print(report, end="")


if __name__ == "__main__":
    main()
