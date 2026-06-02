#!/usr/bin/env python3
"""Run zero-shot or few-shot LLM baselines on a JSONL sample."""

from argparse import ArgumentParser
import os
from random import Random
from pathlib import Path
import sys
import time

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
    parser.add_argument("--resume", choices=["true", "false"], default="false")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-initial-sleep", type=float, default=20.0)
    parser.add_argument("--retry-max-sleep", type=float, default=300.0)
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)
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
    output_path = Path(args.output)
    outputs_by_key = _existing_outputs(output_path) if args.resume == "true" else {}
    skipped = 0
    for example in rows:
        key = _example_key(example)
        if _is_success(outputs_by_key.get(key)):
            skipped += 1
            continue
        prompt = (
            build_zero_shot_prompt(example)
            if args.mode == "zero_shot"
            else build_few_shot_prompt(example, train_rows)
        )
        if args.print_first_prompt == "true" and not outputs_by_key:
            print(prompt)
        response = _run_with_retries(
            runner=runner,
            prompt=prompt,
            max_retries=args.max_retries,
            initial_sleep=args.retry_initial_sleep,
            max_sleep=args.retry_max_sleep,
        )
        outputs_by_key[key] = _prediction_row(example, prompt, response)
        _write_ordered_outputs(rows, outputs_by_key, output_path)
        if args.sleep_between_calls > 0 and not response.get("transport_error"):
            time.sleep(args.sleep_between_calls)
    _write_ordered_outputs(rows, outputs_by_key, output_path)
    print(f"Wrote {len(outputs_by_key)} LLM baseline predictions to {args.output}; skipped {skipped} existing successes")


def _sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size >= len(rows):
        return list(rows)
    rng = Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def _run_with_retries(
    runner: LLMRunner,
    prompt: str,
    max_retries: int,
    initial_sleep: float,
    max_sleep: float,
) -> dict:
    delay = initial_sleep
    last_response = None
    for attempt in range(max_retries + 1):
        try:
            return runner.run(prompt)
        except Exception as error:  # Continue the controlled sample and preserve failures for audit.
            last_response = {
                "raw_response": "",
                "parsed_prediction": None,
                "parse_error": None,
                "transport_error": f"{type(error).__name__}: {error}",
            }
            if attempt >= max_retries or not _is_retryable_transport_error(last_response["transport_error"]):
                return last_response
            time.sleep(delay)
            delay = min(delay * 2, max_sleep)
    return last_response or {
        "raw_response": "",
        "parsed_prediction": None,
        "parse_error": None,
        "transport_error": "unknown retry failure",
    }


def _is_retryable_transport_error(message: str) -> bool:
    lowered = message.lower()
    return "429" in message or "too many requests" in lowered or "timeout" in lowered


def _prediction_row(example: dict, prompt: str, response: dict) -> dict:
    return {
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


def _existing_outputs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {_row_key(row): row for row in read_jsonl(path)}


def _write_ordered_outputs(rows: list[dict], outputs_by_key: dict[str, dict], path: Path) -> None:
    ordered = [outputs_by_key[_example_key(row)] for row in rows if _example_key(row) in outputs_by_key]
    write_jsonl(ordered, path)


def _is_success(row: dict | None) -> bool:
    return bool(row and not row.get("transport_error"))


def _example_key(example: dict) -> str:
    return example.get("stage8_id") or f"{example.get('sample_group') or example.get('split')}:{example['id']}"


def _row_key(row: dict) -> str:
    return row.get("stage8_id") or f"{row.get('source_split')}:{row['id']}"


if __name__ == "__main__":
    main()
