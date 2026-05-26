#!/usr/bin/env python3
"""Audit a generated pilot semantic API diagnosis dataset."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.audit import audit_dataset_file


def main() -> None:
    parser = ArgumentParser(description="Audit a pilot semantic API diagnosis JSONL dataset.")
    parser.add_argument("path", help="Path to the generated JSONL dataset.")
    parser.add_argument("--samples-per-group", type=int, default=2)
    parser.add_argument("--show-full-input", choices=["true", "false"], default="false")
    parser.add_argument("--output", default="data/generated/pilot_audit_report.txt")
    args = parser.parse_args()

    report, _warnings = audit_dataset_file(
        args.path,
        output_path=args.output,
        samples_per_group=args.samples_per_group,
        show_full_input=args.show_full_input == "true",
    )
    print(report, end="")


if __name__ == "__main__":
    main()
