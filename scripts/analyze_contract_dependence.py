#!/usr/bin/env python3
"""Analyze whether examples require endpoint contract information."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.evaluation.contract_dependence import analyze_contract_dependence


def main() -> None:
    parser = ArgumentParser(description="Analyze request-obvious vs contract-dependent examples.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--seen-test", required=True)
    parser.add_argument("--unseen-test", required=True)
    parser.add_argument("--output", default="data/generated/contract_dependence_report.md")
    args = parser.parse_args()

    result = analyze_contract_dependence(args.train, args.seen_test, args.unseen_test, args.output)
    print(f"Wrote contract dependence report to {args.output}")
    print("warnings:")
    for warning in result["warnings"] or ["none"]:
        print(f"- {warning}")


if __name__ == "__main__":
    main()
