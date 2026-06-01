#!/usr/bin/env python3
"""Run zero-shot or few-shot LLM baselines on a JSONL sample."""

from argparse import ArgumentParser
import os
from random import Random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_api_diagnosis.llm_baselines.prompts import build_few_shot_prompt, build_zero_shot_prompt
from semantic_api_diagnosis.llm_baselines.runner import LLMRunner
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = ArgumentParser(description="Run an LLM baseline on a generated dataset sample.")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], required=True)
    parser.add_argument("--provider", choices=["mock", "openai", "manual"], default="mock")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--dry-run", choices=["true", "false"], default="false")
    parser.add_argument("--print-first-prompt", choices=["true", "false"], default="false")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    rows = _sample(read_jsonl(args.input), args.sample_size, args.seed)
    train_rows = read_jsonl(args.train) if args.train else []
    if args.mode == "few_shot" and not train_rows:
        parser.error("--train is required for few_shot mode")
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set for provider=openai")
    runner = LLMRunner(
        provider=args.provider,
        model=args.model,
        dry_run=args.dry_run == "true",
        timeout=args.timeout,
    )
    outputs = []
    for example in rows:
        prompt = (
            build_zero_shot_prompt(example)
            if args.mode == "zero_shot"
            else build_few_shot_prompt(example, train_rows)
        )
        if args.print_first_prompt == "true" and not outputs:
            print(prompt)
        try:
            response = runner.run(prompt)
        except Exception as error:  # Continue the controlled sample and preserve failures for audit.
            response = {
                "raw_response": "",
                "parsed_prediction": None,
                "parse_error": None,
                "transport_error": f"{type(error).__name__}: {error}",
            }
        outputs.append(
            {
                "id": example["id"],
                "stage8_id": example.get("stage8_id"),
                "source_split": example.get("sample_group") or example.get("split"),
                "endpoint_family": example.get("endpoint_family"),
                "prompt": prompt,
                "raw_response": response["raw_response"],
                "parsed_prediction": response["parsed_prediction"],
                "gold": example["target"],
                "parse_error": response["parse_error"],
                "transport_error": response.get("transport_error"),
                "provider_response_id": response.get("provider_response_id"),
                "usage": response.get("usage"),
            }
        )
    write_jsonl(outputs, args.output)
    print(f"Wrote {len(outputs)} LLM baseline predictions to {args.output}")


def _sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size >= len(rows):
        return list(rows)
    rng = Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


if __name__ == "__main__":
    main()
