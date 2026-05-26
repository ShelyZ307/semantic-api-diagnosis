#!/usr/bin/env python3
"""Print the Version 1 dataset split and family plan."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.dataset_plan import dataset_plan_text


def main() -> None:
    print(dataset_plan_text())


if __name__ == "__main__":
    main()
