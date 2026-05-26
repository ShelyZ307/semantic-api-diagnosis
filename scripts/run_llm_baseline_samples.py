#!/usr/bin/env python3
"""Run Stage 6A LLM baseline samples for full and hard subsets."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


SAMPLES = [
    ("full_seen_sample_100", "data/generated/final_seen_test.jsonl", 100, "zero_shot"),
    ("full_unseen_sample_100", "data/generated/final_unseen_family_test.jsonl", 100, "zero_shot"),
    ("contract_dependent_seen_sample_50", "data/generated/contract_dependent_seen_test.jsonl", 50, "few_shot"),
    ("contract_dependent_unseen_sample_50", "data/generated/contract_dependent_unseen_test.jsonl", 50, "few_shot"),
]


def main() -> None:
    parser = ArgumentParser(description="Run Stage 6A LLM baseline sample jobs.")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "manual"])
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--train", default="data/generated/final_train.jsonl")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--dry-run", choices=["true", "false"], default="false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, input_path, sample_size, mode) in enumerate(SAMPLES):
        prediction_path = output_dir / f"llm_{name}_{args.provider}_predictions.jsonl"
        result_path = output_dir / f"llm_{name}_{args.provider}_results.json"
        command = [
            sys.executable,
            "scripts/run_llm_baseline.py",
            "--mode",
            mode,
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--input",
            input_path,
            "--sample-size",
            str(sample_size),
            "--output",
            str(prediction_path),
            "--seed",
            str(args.seed + index),
            "--dry-run",
            args.dry_run,
        ]
        if mode == "few_shot":
            command.extend(["--train", args.train])
        subprocess.run(command, check=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_llm_predictions.py",
                "--predictions",
                str(prediction_path),
                "--output",
                str(result_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
