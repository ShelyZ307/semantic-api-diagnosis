#!/usr/bin/env python3
"""Run placeholder EDA over a JSONL dataset."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.eda.summarize import summarize_examples
from semantic_api_diagnosis.serialization.jsonl import read_jsonl


def main() -> None:
    parser = ArgumentParser(description="Summarize a generated JSONL dataset.")
    parser.add_argument("path", nargs="?", default="data/generated/sample.jsonl")
    args = parser.parse_args()

    examples = read_jsonl(args.path)
    print(json.dumps(summarize_examples(examples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
