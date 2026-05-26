#!/usr/bin/env python3
"""Create shortcut diagnostic views from a generated dataset."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.quality_gate import create_shortcut_views


def main() -> None:
    parser = ArgumentParser(description="Create shortcut dataset views.")
    parser.add_argument("path")
    parser.add_argument("--output-dir", default="data/generated/shortcut_views")
    args = parser.parse_args()

    paths = create_shortcut_views(args.path, args.output_dir)
    for view_name, path in sorted(paths.items()):
        print(f"{view_name}: {path}")


if __name__ == "__main__":
    main()
