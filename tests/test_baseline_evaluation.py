import inspect
import json
import subprocess
import sys

from semantic_api_diagnosis.baselines.family_frequency import FamilyFrequencyBaseline
from semantic_api_diagnosis.baselines.majority import MajorityBaseline
from semantic_api_diagnosis.baselines.visible_rule_based import VisibleRuleBasedBaseline
import semantic_api_diagnosis.baselines.visible_rule_based as visible_rule_based
from semantic_api_diagnosis.data_generation.generate import generate_examples
from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.serialization.jsonl import write_jsonl


def test_metrics_compute_on_toy_multilabel_example() -> None:
    gold = [
        {
            "target": {
                "error_labels": ["missing_authentication", "semantic_state_violation"],
                "validity": "invalid",
                "severity_bucket": "high",
            }
        },
        {
            "target": {
                "error_labels": [],
                "validity": "valid",
                "severity_bucket": "none",
            }
        },
    ]
    predictions = [
        {
            "error_labels": ["missing_authentication"],
            "validity": "invalid",
            "severity_bucket": "high",
        },
        {
            "error_labels": [],
            "validity": "valid",
            "severity_bucket": "none",
        },
    ]

    metrics = evaluate_predictions(gold, predictions)

    assert metrics["all_labels"]["micro_precision"] == 1.0
    assert metrics["all_labels"]["micro_recall"] == 0.5
    assert metrics["exact_match_label_set_accuracy"] == 0.5
    assert metrics["critical_semantic_error_miss_rate"] == 1.0
    assert metrics["validity"]["accuracy"] == 1.0
    assert metrics["severity_bucket"]["accuracy"] == 1.0


def test_majority_baseline_runs() -> None:
    train = generate_examples(100, split="train", family_mode="seen", seed=101)
    model = MajorityBaseline(label_threshold=None).fit(train)

    predictions = model.predict(train[:5])

    assert len(predictions) == 5
    assert all(prediction["error_labels"] == [] for prediction in predictions)


def test_family_frequency_baseline_handles_unseen_family_fallback() -> None:
    train = generate_examples(100, split="train", family_mode="seen", seed=101)
    unseen = generate_examples(20, split="unseen_family_test", family_mode="unseen", seed=104)
    model = FamilyFrequencyBaseline(label_threshold=0.3).fit(train)

    prediction = model.predict_one(unseen[0])

    assert set(prediction) == {"error_labels", "validity", "severity_bucket"}


def test_visible_rule_based_baseline_does_not_import_hidden_validators() -> None:
    source = inspect.getsource(visible_rule_based)

    assert "hidden_validators" not in source
    assert "validate_example" not in source


def test_visible_rule_based_baseline_detects_obvious_authentication_error() -> None:
    example = generate_examples(20, split="train", family_mode="seen", seed=101)[10]
    example["serialized_input"] = example["serialized_input"].replace(
        "Authentication: provided",
        "Authentication: missing",
    )

    prediction = VisibleRuleBasedBaseline().fit([]).predict_one(example)

    assert "missing_authentication" in prediction["error_labels"]


def test_evaluator_script_runs_and_writes_json_and_markdown(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    seen_path = tmp_path / "seen_test.jsonl"
    unseen_path = tmp_path / "unseen_test.jsonl"
    output_path = tmp_path / "baseline_results.json"
    write_jsonl(generate_examples(100, split="train", family_mode="seen", seed=101), train_path)
    write_jsonl(generate_examples(50, split="validation", family_mode="seen", seed=102), validation_path)
    write_jsonl(generate_examples(50, split="seen_test", family_mode="seen", seed=103), seen_path)
    write_jsonl(generate_examples(50, split="unseen_family_test", family_mode="unseen", seed=104), unseen_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baselines.py",
            "--train",
            str(train_path),
            "--validation",
            str(validation_path),
            "--seen-test",
            str(seen_path),
            "--unseen-test",
            str(unseen_path),
            "--output",
            str(output_path),
        ],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        check=True,
        capture_output=True,
        text=True,
    )

    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()
    assert "majority_empty" in results["splits"]["validation"]
    assert "visible_rule_based" in results["splits"]["seen_test"]
    assert results["shortcut_views"]


def test_evaluator_includes_hard_subset_sections_when_files_exist(tmp_path) -> None:
    paths = _write_small_eval_files(tmp_path)
    hard_seen = tmp_path / "contract_dependent_seen_test.jsonl"
    hard_unseen = tmp_path / "contract_dependent_unseen_test.jsonl"
    write_jsonl(generate_examples(30, split="seen_test", family_mode="seen", seed=203), hard_seen)
    write_jsonl(generate_examples(30, split="unseen_family_test", family_mode="unseen", seed=204), hard_unseen)
    output_path = tmp_path / "baseline_results.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baselines.py",
            "--train",
            str(paths["train"]),
            "--validation",
            str(paths["validation"]),
            "--seen-test",
            str(paths["seen"]),
            "--unseen-test",
            str(paths["unseen"]),
            "--output",
            str(output_path),
        ],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        check=True,
        capture_output=True,
        text=True,
    )

    markdown = output_path.with_suffix(".md").read_text(encoding="utf-8")
    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert "contract_dependent_seen_test" in results["hard_subsets"]
    assert "Hard Contract-Dependent Subset Results" in markdown
    assert "Contract-Dependence Summary" in markdown
    assert "Future model results should be reported both" in markdown


def test_evaluator_missing_hard_subsets_message_is_clear(tmp_path) -> None:
    paths = _write_small_eval_files(tmp_path)
    output_path = tmp_path / "nested" / "baseline_results.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baselines.py",
            "--train",
            str(paths["train"]),
            "--validation",
            str(paths["validation"]),
            "--seen-test",
            str(paths["seen"]),
            "--unseen-test",
            str(paths["unseen"]),
            "--output",
            str(output_path),
        ],
        cwd="/Users/shelyz/Documents/semantic-api-diagnosis",
        check=True,
        capture_output=True,
        text=True,
    )

    markdown = output_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Run scripts/analyze_contract_dependence.py first" in result.stdout
    assert "Missing Hard Subsets" in markdown
    assert "Run scripts/analyze_contract_dependence.py first" in markdown


def _write_small_eval_files(tmp_path) -> dict:
    paths = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "seen": tmp_path / "seen_test.jsonl",
        "unseen": tmp_path / "unseen_test.jsonl",
    }
    write_jsonl(generate_examples(100, split="train", family_mode="seen", seed=101), paths["train"])
    write_jsonl(generate_examples(50, split="validation", family_mode="seen", seed=102), paths["validation"])
    write_jsonl(generate_examples(50, split="seen_test", family_mode="seen", seed=103), paths["seen"])
    write_jsonl(generate_examples(50, split="unseen_family_test", family_mode="unseen", seed=104), paths["unseen"])
    return paths
