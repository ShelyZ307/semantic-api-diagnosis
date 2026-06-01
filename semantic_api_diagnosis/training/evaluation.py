"""Threshold tuning and canonical evaluation for fine-tuned encoders."""

from __future__ import annotations

from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.labels import severity_bucket
from semantic_api_diagnosis.training.labels import ERROR_LABELS, labels_to_multihot, multihot_to_labels
from semantic_api_diagnosis.training.metrics import compute_multilabel_metrics


LABEL_SEVERITIES = {
    "missing_required_field": "high",
    "wrong_type": "medium",
    "invalid_value_range": "medium",
    "malformed_url": "medium",
    "wrong_http_method": "high",
    "missing_authentication": "high",
    "unexpected_or_malformed_body_structure": "high",
    "semantic_cross_field_violation": "high",
    "semantic_domain_constraint_violation": "high",
    "semantic_state_violation": "high",
}

DEFAULT_THRESHOLD_GRID = [round(value / 20, 2) for value in range(1, 20)]


def threshold_vector(thresholds: float | dict[str, float]) -> list[float]:
    if isinstance(thresholds, (int, float)):
        return [float(thresholds)] * len(ERROR_LABELS)
    return [float(thresholds[label]) for label in ERROR_LABELS]


def tune_thresholds(
    rows: list[dict],
    mode: str = "per_label",
    grid: list[float] | None = None,
) -> dict:
    if mode not in {"global", "per_label"}:
        raise ValueError("threshold tuning mode must be global or per_label")
    grid = grid or DEFAULT_THRESHOLD_GRID
    y_true, y_scores = _vectors(rows)
    if mode == "global":
        selected = max(grid, key=lambda value: (_score(y_true, y_scores, value), -abs(value - 0.5)))
        thresholds = {label: selected for label in ERROR_LABELS}
    else:
        thresholds = {}
        for index, label in enumerate(ERROR_LABELS):
            thresholds[label] = max(
                grid,
                key=lambda value: (
                    _column_f1(y_true, y_scores, index, value),
                    -abs(value - 0.5),
                ),
            )
    return {
        "method": mode,
        "grid": grid,
        "thresholds": thresholds,
        "default_0_5_metrics": compute_metrics_for_prediction_rows(rows, 0.5),
        "tuned_metrics": compute_metrics_for_prediction_rows(rows, thresholds),
    }


def apply_thresholds(scores: list[float], thresholds: float | dict[str, float]) -> list[str]:
    return multihot_to_labels(scores, threshold=threshold_vector(thresholds))


def prediction_from_labels(labels: list[str]) -> dict:
    severities = [LABEL_SEVERITIES[label] for label in labels]
    return {
        "error_labels": labels,
        "validity": "invalid" if labels else "valid",
        "severity_bucket": severity_bucket(severities),
    }


def compute_metrics_for_prediction_rows(
    rows: list[dict],
    thresholds: float | dict[str, float] = 0.5,
) -> dict:
    predictions = []
    gold = []
    for row in rows:
        labels = apply_thresholds([row["scores"][label] for label in ERROR_LABELS], thresholds)
        predictions.append(prediction_from_labels(labels))
        gold.append({"target": _row_gold_target(row)})
    metrics = evaluate_predictions(gold, predictions)
    metrics["micro_f1"] = metrics["all_labels"]["micro_f1"]
    metrics["macro_f1"] = metrics["all_labels"]["macro_f1"]
    metrics["exact_match"] = metrics["exact_match_label_set_accuracy"]
    metrics["semantic_micro_f1"] = metrics["semantic_labels"]["micro_f1"]
    metrics["semantic_macro_f1"] = metrics["semantic_labels"]["macro_f1"]
    return metrics


def add_thresholded_prediction(row: dict, thresholds: float | dict[str, float]) -> dict:
    updated = dict(row)
    labels = apply_thresholds([row["scores"][label] for label in ERROR_LABELS], thresholds)
    updated["predicted_error_labels"] = labels
    updated["predicted_target"] = prediction_from_labels(labels)
    return updated


def _vectors(rows: list[dict]) -> tuple[list[list[int]], list[list[float]]]:
    y_true = [labels_to_multihot(_row_gold_target(row)["error_labels"]) for row in rows]
    y_scores = [[row["scores"][label] for label in ERROR_LABELS] for row in rows]
    return y_true, y_scores


def _row_gold_target(row: dict) -> dict:
    if row.get("gold_target") is not None:
        return row["gold_target"]
    labels = row.get("gold_error_labels", [])
    return {
        "error_labels": labels,
        "validity": "invalid" if labels else "valid",
        "severity_bucket": severity_bucket([LABEL_SEVERITIES[label] for label in labels]),
    }


def _score(y_true: list[list[int]], y_scores: list[list[float]], threshold: float) -> float:
    return compute_multilabel_metrics(y_true, y_scores, threshold=threshold)["macro_f1"]


def _column_f1(y_true: list[list[int]], y_scores: list[list[float]], index: int, threshold: float) -> float:
    tp = fp = fn = 0
    for truth, scores in zip(y_true, y_scores):
        predicted = scores[index] >= threshold
        if truth[index] and predicted:
            tp += 1
        elif not truth[index] and predicted:
            fp += 1
        elif truth[index] and not predicted:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
