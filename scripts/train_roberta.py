#!/usr/bin/env python3
"""RoBERTa training placeholder script."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.training.train_roberta import main


if __name__ == "__main__":
    main()
