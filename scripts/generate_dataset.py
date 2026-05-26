#!/usr/bin/env python3
"""Generate a placeholder dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.data_generation.generate import main


if __name__ == "__main__":
    main()
