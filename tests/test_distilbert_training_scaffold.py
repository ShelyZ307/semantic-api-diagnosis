import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from semantic_api_diagnosis.training.train_distilbert import limit_records, _build_training_arguments


def test_limit_records_uses_deterministic_first_n() -> None:
    records = [{"id": index} for index in range(5)]

    assert limit_records(records, None) == records
    assert limit_records(records, 2) == [{"id": 0}, {"id": 1}]


def test_training_arguments_accept_stage_6b_options(tmp_path) -> None:
    class DummyTrainingArguments:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            learning_rate,
            weight_decay,
            logging_steps,
            save_strategy,
            seed,
            report_to,
            eval_strategy,
        ):
            self.values = locals()

    args = argparse.Namespace(
        epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        seed=42,
        eval_strategy="epoch",
    )

    built = _build_training_arguments(DummyTrainingArguments, args, tmp_path)

    assert built.values["output_dir"] == str(tmp_path)
    assert built.values["per_device_train_batch_size"] == 4
    assert built.values["eval_strategy"] == "epoch"


def test_prediction_row_formatting_and_metrics() -> None:
    module = _load_evaluation_script()
    example = {
        "id": "ex_1",
        "serialized_input": "text",
        "target": {"error_labels": ["missing_required_field"]},
    }
    scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    row = module.format_prediction_row(example, scores, threshold=0.5)
    metrics = module.compute_metrics_for_prediction_rows([row])

    assert row["id"] == "ex_1"
    assert row["gold_error_labels"] == ["missing_required_field"]
    assert row["predicted_error_labels"] == ["missing_required_field"]
    assert row["scores"]["missing_required_field"] == 0.9
    assert metrics["exact_match"] == 1.0


def test_evaluate_finetuned_model_fails_clearly_for_missing_model(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        '{"id":"ex_1","serialized_input":"text","target":{"error_labels":[]}}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_finetuned_model.py",
            "--model-dir",
            str(tmp_path / "missing_model"),
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
        ],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No trained model found" in result.stderr


def _load_evaluation_script():
    path = Path("/Users/shelyz/Documents/semantic-api-diagnosis/scripts/evaluate_finetuned_model.py")
    spec = importlib.util.spec_from_file_location("evaluate_finetuned_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
