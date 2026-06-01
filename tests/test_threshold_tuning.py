from semantic_api_diagnosis.training.evaluation import (
    add_thresholded_prediction,
    compute_metrics_for_prediction_rows,
    tune_thresholds,
)
from semantic_api_diagnosis.training.labels import ERROR_LABELS


def test_per_label_threshold_tuning_uses_validation_scores() -> None:
    rows = [
        _row(["missing_required_field"], {"missing_required_field": 0.35}),
        _row([], {"missing_required_field": 0.20}),
        _row(["semantic_state_violation"], {"semantic_state_violation": 0.45}),
        _row([], {"semantic_state_violation": 0.25}),
    ]

    tuned = tune_thresholds(rows, mode="per_label", grid=[0.2, 0.3, 0.4, 0.5])

    assert tuned["thresholds"]["missing_required_field"] == 0.3
    assert tuned["thresholds"]["semantic_state_violation"] == 0.4
    assert tuned["tuned_metrics"]["macro_f1"] > tuned["default_0_5_metrics"]["macro_f1"]


def test_canonical_metrics_include_required_auxiliary_outputs() -> None:
    rows = [_row(["semantic_state_violation"], {"semantic_state_violation": 0.9})]

    metrics = compute_metrics_for_prediction_rows(rows, 0.5)

    assert metrics["all_labels"]["micro_f1"] == 1.0
    assert metrics["semantic_labels"]["macro_f1"] > 0.0
    assert metrics["critical_semantic_error_miss_rate"] == 0.0
    assert metrics["validity"]["accuracy"] == 1.0
    assert metrics["severity_bucket"]["accuracy"] == 1.0


def test_thresholded_prediction_contains_derived_target() -> None:
    row = _row([], {"wrong_type": 0.8})

    prediction = add_thresholded_prediction(row, 0.5)

    assert prediction["predicted_error_labels"] == ["wrong_type"]
    assert prediction["predicted_target"]["validity"] == "invalid"
    assert prediction["predicted_target"]["severity_bucket"] == "medium"


def _row(labels: list[str], overrides: dict[str, float]) -> dict:
    scores = {label: 0.01 for label in ERROR_LABELS}
    scores.update(overrides)
    return {
        "id": "example",
        "gold_target": {
            "error_labels": labels,
            "validity": "invalid" if labels else "valid",
            "severity_bucket": "high" if labels else "none",
        },
        "scores": scores,
    }
