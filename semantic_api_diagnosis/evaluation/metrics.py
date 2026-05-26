"""Metrics for multi-label semantic API diagnosis evaluation."""

from collections import Counter

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY, SEMANTIC_LABELS


LABELS = sorted(FIXED_LABEL_TAXONOMY)
SEMANTIC = sorted(SEMANTIC_LABELS)
SEVERITY_BUCKETS = ["none", "low", "medium", "high"]


def evaluate_predictions(gold: list[dict], predictions: list[dict]) -> dict:
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have the same length")
    gold_labels = [set(example["target"]["error_labels"]) for example in gold]
    predicted_labels = [set(prediction.get("error_labels", [])) for prediction in predictions]
    label_metrics = _multilabel_metrics(gold_labels, predicted_labels, LABELS)
    semantic_metrics = _multilabel_metrics(gold_labels, predicted_labels, SEMANTIC)
    validity_gold = [example["target"]["validity"] for example in gold]
    validity_pred = [prediction.get("validity", "valid") for prediction in predictions]
    severity_gold = [example["target"]["severity_bucket"] for example in gold]
    severity_pred = [prediction.get("severity_bucket", "none") for prediction in predictions]
    return {
        "all_labels": label_metrics,
        "semantic_labels": {
            "micro_f1": semantic_metrics["micro_f1"],
            "macro_f1": semantic_metrics["macro_f1"],
            "per_label": semantic_metrics["per_label"],
        },
        "exact_match_label_set_accuracy": _exact_match(gold_labels, predicted_labels),
        "validity": _binary_invalid_metrics(validity_gold, validity_pred),
        "severity_bucket": _multiclass_metrics(severity_gold, severity_pred, SEVERITY_BUCKETS),
        "critical_semantic_error_miss_rate": _semantic_miss_rate(gold, predicted_labels, high_only=False),
        "critical_semantic_error_miss_rate_high_severity": _semantic_miss_rate(
            gold,
            predicted_labels,
            high_only=True,
        ),
    }


def _multilabel_metrics(gold_sets: list[set[str]], predicted_sets: list[set[str]], labels: list[str]) -> dict:
    per_label = {}
    totals = Counter()
    for label in labels:
        tp = fp = fn = 0
        for gold, predicted in zip(gold_sets, predicted_sets):
            if label in gold and label in predicted:
                tp += 1
            elif label not in gold and label in predicted:
                fp += 1
            elif label in gold and label not in predicted:
                fn += 1
        metrics = _prf(tp, fp, fn)
        per_label[label] = metrics
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    micro = _prf(totals["tp"], totals["fp"], totals["fn"])
    macro_precision = _mean([per_label[label]["precision"] for label in labels])
    macro_recall = _mean([per_label[label]["recall"] for label in labels])
    macro_f1 = _mean([per_label[label]["f1"] for label in labels])
    return {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def _binary_invalid_metrics(gold: list[str], predicted: list[str]) -> dict:
    tp = fp = fn = 0
    for gold_value, predicted_value in zip(gold, predicted):
        if gold_value == "invalid" and predicted_value == "invalid":
            tp += 1
        elif gold_value != "invalid" and predicted_value == "invalid":
            fp += 1
        elif gold_value == "invalid" and predicted_value != "invalid":
            fn += 1
    metrics = _prf(tp, fp, fn)
    metrics["accuracy"] = _accuracy(gold, predicted)
    return metrics


def _multiclass_metrics(gold: list[str], predicted: list[str], labels: list[str]) -> dict:
    per_label = {}
    for label in labels:
        tp = fp = fn = 0
        for gold_value, predicted_value in zip(gold, predicted):
            if gold_value == label and predicted_value == label:
                tp += 1
            elif gold_value != label and predicted_value == label:
                fp += 1
            elif gold_value == label and predicted_value != label:
                fn += 1
        per_label[label] = _prf(tp, fp, fn)
    return {
        "accuracy": _accuracy(gold, predicted),
        "macro_f1": _mean([metrics["f1"] for metrics in per_label.values()]),
        "per_label": per_label,
    }


def _semantic_miss_rate(gold: list[dict], predicted_labels: list[set[str]], high_only: bool) -> float | None:
    missed = 0
    total = 0
    for example, predicted in zip(gold, predicted_labels):
        if high_only and example["target"].get("severity_bucket") != "high":
            continue
        gold_semantic = set(example["target"]["error_labels"]) & SEMANTIC_LABELS
        if not gold_semantic:
            continue
        total += 1
        if not (predicted & SEMANTIC_LABELS):
            missed += 1
    return None if total == 0 else missed / total


def _exact_match(gold_sets: list[set[str]], predicted_sets: list[set[str]]) -> float:
    if not gold_sets:
        return 0.0
    return sum(gold == predicted for gold, predicted in zip(gold_sets, predicted_sets)) / len(gold_sets)


def _accuracy(gold: list[str], predicted: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(gold_value == predicted_value for gold_value, predicted_value in zip(gold, predicted)) / len(gold)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
