"""Model builders for encoder fine-tuning."""


def build_sequence_classifier(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 10,
):
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Transformers is required for model training. Install with: "
            'python3 -m pip install -e ".[train]"'
        ) from exc
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
