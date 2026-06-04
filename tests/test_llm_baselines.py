import json
import subprocess
import sys
from pathlib import Path

from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.dataset_plan import PLANNED_UNSEEN_ENDPOINT_FAMILIES
from semantic_api_diagnosis.llm_baselines.prompts import (
    FINAL_JSON_ONLY_INSTRUCTION,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    select_few_shot_examples,
)
from semantic_api_diagnosis.llm_baselines.runner import LLMRunner, _response_output_text
from semantic_api_diagnosis.serialization.jsonl import read_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_zero_shot_prompt_contains_taxonomy_and_serialized_input() -> None:
    example = generate_examples(1, split="seen_test", family_mode="seen", seed=201)[0]
    prompt = build_zero_shot_prompt(example)

    assert "missing_required_field" in prompt
    assert "semantic_state_violation" in prompt
    assert example["serialized_input"] in prompt
    assert "Return strict JSON only" in prompt


def test_zero_shot_prompt_ends_with_final_json_only_instruction() -> None:
    example = generate_examples(1, split="seen_test", family_mode="seen", seed=201)[0]
    prompt = build_zero_shot_prompt(example)

    assert prompt.endswith(FINAL_JSON_ONLY_INSTRUCTION)


def test_few_shot_prompt_uses_only_training_seen_examples() -> None:
    train = generate_examples(80, split="train", family_mode="seen", seed=101)
    unseen_target = generate_examples(1, split="unseen_family_test", family_mode="unseen", seed=202)[0]
    prompt = build_few_shot_prompt(unseen_target, train)
    demos = select_few_shot_examples(train)

    assert demos
    assert all(example["split"] == "train" for example in demos)
    assert all(example["endpoint_family"] not in PLANNED_UNSEEN_ENDPOINT_FAMILIES for example in demos)
    assert "Demonstration 1" in prompt
    assert "Now diagnose this request. Use the same JSON schema." in prompt
    assert unseen_target["serialized_input"] in prompt


def test_few_shot_prompt_ends_with_final_json_only_instruction_and_contains_taxonomy() -> None:
    train = generate_examples(80, split="train", family_mode="seen", seed=101)
    unseen_target = generate_examples(1, split="unseen_family_test", family_mode="unseen", seed=202)[0]
    prompt = build_few_shot_prompt(unseen_target, train)

    assert prompt.endswith(FINAL_JSON_ONLY_INSTRUCTION)
    assert "missing_required_field" in prompt
    assert "semantic_state_violation" in prompt


def test_mock_runner_returns_parseable_json() -> None:
    runner = LLMRunner(provider="mock", model="mock-model")
    result = runner.run("Serialized input:\nRequest:\nAuthentication: missing\nMethod: POST\nURL: /ok")

    assert result["parse_error"] is None
    assert result["parsed_prediction"]["validity"] == "invalid"
    assert "missing_authentication" in result["parsed_prediction"]["error_labels"]


def test_openai_response_output_text_extracts_structured_output() -> None:
    body = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"validity":"valid","error_labels":[],"severity_bucket":"none"}',
                    }
                ]
            }
        ]
    }

    assert _response_output_text(body).startswith('{"validity"')


def test_openai_runner_requests_strict_structured_output(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "id": "resp_test",
                "output_text": '{"validity":"valid","error_labels":[],"severity_bucket":"none"}',
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("httpx.post", fake_post)

    result = LLMRunner(provider="openai", model="gpt-4o-mini").run("prompt")

    assert result["parsed_prediction"]["validity"] == "valid"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["json"]["text"]["format"]["type"] == "json_schema"
    assert captured["json"]["text"]["format"]["strict"] is True


def test_run_llm_baseline_script_mock_mode_on_tiny_file(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(generate_examples(5, split="seen_test", family_mode="seen", seed=201), input_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_llm_baseline.py",
            "--mode",
            "zero_shot",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--input",
            str(input_path),
            "--sample-size",
            "3",
            "--output",
            str(output_path),
            "--seed",
            "201",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = read_jsonl(output_path)
    assert len(rows) == 3
    assert all("prompt" in row for row in rows)
    assert all(row["parsed_prediction"] for row in rows)


def test_evaluate_llm_predictions_reports_parse_error_rate(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "results.json"
    example = generate_examples(1, split="seen_test", family_mode="seen", seed=201)[0]
    write_jsonl(
        [
            {
                "id": example["id"],
                "source_split": example["split"],
                "prompt": "prompt",
                "raw_response": "{}",
                "parsed_prediction": None,
                "gold": example["target"],
                "parse_error": "bad json",
                "transport_error": "provider unavailable",
            }
        ],
        prediction_path,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_llm_predictions.py",
            "--predictions",
            str(prediction_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert results["parse_error_rate"] == 1.0
    assert results["invalid_json_rate"] == 0.0
    assert results["transport_error_rate"] == 1.0
    assert output_path.with_suffix(".md").exists()
