#!/usr/bin/env python3
"""Run controlled provider-backed Stage 7 zero-shot and few-shot LLM samples."""

from __future__ import annotations

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys


VIEWS = [
    ("seen", "data/generated/final_seen_test.jsonl"),
    ("unseen", "data/generated/final_unseen_family_test.jsonl"),
    ("contract_dependent_unseen", "data/generated/contract_dependent_unseen_test.jsonl"),
]


def main() -> None:
    parser = ArgumentParser(description="Run controlled Stage 7 provider-backed LLM baselines.")
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--train", default="data/generated/final_train.jsonl")
    parser.add_argument("--output-dir", default="data/generated/stage7_llm")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set for provider=openai")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode_index, mode in enumerate(["zero_shot", "few_shot"]):
        for view_index, (view, input_path) in enumerate(VIEWS):
            stem = f"{mode}_{view}"
            predictions = output_dir / f"{stem}_predictions.jsonl"
            results = output_dir / f"{stem}_results.json"
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
                str(args.sample_size),
                "--output",
                str(predictions),
                "--seed",
                str(args.seed + mode_index * len(VIEWS) + view_index),
                "--timeout",
                str(args.timeout),
            ]
            if mode == "few_shot":
                command.extend(["--train", args.train])
            subprocess.run(command, check=True)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_llm_predictions.py",
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(results),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
