"""Training-facing multi-label metric helpers."""

from __future__ import annotations

from semantic_api_diagnosis.training.labels import ERROR_LABELS

SEMANTIC_LABELS = [
    "semantic_cross_field_violation",
    "semantic_domain_constraint_violation",
    "semantic_state_violation",
]


def compute_multilabel_metrics(y_true, y_pred, threshold: float = 0.5) -> dict:
    true_vectors = [_as_binary_vector(row, threshold=0.5) for row in y_true]
    pred_vectors = [_as_binary_vector(row, threshold=threshold) for row in y_pred]
    if len(true_vectors) != len(pred_vectors):
        raise ValueError("y_true and y_pred must have the same number of rows")
    if any(len(row) != len(ERROR_LABELS) for row in true_vectors + pred_vectors):
        raise ValueError(f"All vectors must have length {len(ERROR_LABELS)}")

    per_label = {}
    for index, label in enumerate(ERROR_LABELS):
        per_label[label] = _f1_for_column(true_vectors, pred_vectors, index)
    semantic_indices = [ERROR_LABELS.index(label) for label in SEMANTIC_LABELS]
    semantic_per_label = {
        label: per_label[label]
        for label in SEMANTIC_LABELS
    }
    sklearn_scores = _sklearn_scores_if_available(true_vectors, pred_vectors)
    return {
        "micro_f1": sklearn_scores.get("micro_f1", _micro_f1(true_vectors, pred_vectors, range(len(ERROR_LABELS)))),
        "macro_f1": sklearn_scores.get("macro_f1", _mean([metrics["f1"] for metrics in per_label.values()])),
        "exact_match": _exact_match(true_vectors, pred_vectors),
        "per_label_f1": {label: metrics["f1"] for label, metrics in per_label.items()},
        "semantic_macro_f1": _mean([per_label[label]["f1"] for label in SEMANTIC_LABELS]),
        "semantic_micro_f1": _micro_f1(true_vectors, pred_vectors, semantic_indices),
        "per_semantic_label_f1": {label: metrics["f1"] for label, metrics in semantic_per_label.items()},
    }


def _as_binary_vector(row, threshold: float) -> list[int]:
    return [1 if float(value) >= threshold else 0 for value in list(row)]


def _sklearn_scores_if_available(y_true: list[list[int]], y_pred: list[list[int]]) -> dict:
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return {}
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _f1_for_column(y_true: list[list[int]], y_pred: list[list[int]], index: int) -> dict:
    tp = fp = fn = 0
    for true_row, pred_row in zip(y_true, y_pred):
        if true_row[index] and pred_row[index]:
            tp += 1
        elif not true_row[index] and pred_row[index]:
            fp += 1
        elif true_row[index] and not pred_row[index]:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _micro_f1(y_true: list[list[int]], y_pred: list[list[int]], indices) -> float:
    tp = fp = fn = 0
    for index in indices:
        metrics_counts = _counts_for_column(y_true, y_pred, index)
        tp += metrics_counts["tp"]
        fp += metrics_counts["fp"]
        fn += metrics_counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _counts_for_column(y_true: list[list[int]], y_pred: list[list[int]], index: int) -> dict:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for true_row, pred_row in zip(y_true, y_pred):
        if true_row[index] and pred_row[index]:
            counts["tp"] += 1
        elif not true_row[index] and pred_row[index]:
            counts["fp"] += 1
        elif true_row[index] and not pred_row[index]:
            counts["fn"] += 1
    return counts


def _exact_match(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    if not y_true:
        return 0.0
    return sum(true_row == pred_row for true_row, pred_row in zip(y_true, y_pred)) / len(y_true)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
