#!/usr/bin/env python3
"""Run the baseline rule-based validator on generated examples."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.validators.baseline import validate_example


def main() -> None:
    examples = generate_examples(count=4)
    for example in examples:
        print(validate_example(example))


if __name__ == "__main__":
    main()
