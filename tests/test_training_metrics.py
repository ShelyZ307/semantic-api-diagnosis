from semantic_api_diagnosis.training.metrics import compute_multilabel_metrics


def test_compute_multilabel_metrics_returns_expected_keys() -> None:
    y_true = [
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    y_pred = [
        [0.9, 0.1, 0, 0, 0, 0, 0, 0.8, 0.1, 0.2],
        [0.1, 0.7, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1],
    ]

    metrics = compute_multilabel_metrics(y_true, y_pred)

    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
    assert "exact_match" in metrics
    assert "per_label_f1" in metrics
    assert "semantic_macro_f1" in metrics
    assert metrics["exact_match"] == 1.0
    assert metrics["semantic_macro_f1"] >= 0.0
