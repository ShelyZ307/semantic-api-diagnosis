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


def build_pos_weighted_trainer_class():
    """Return a Trainer subclass using train-split positive-class weights."""
    try:
        import torch
        from transformers import Trainer
    except ImportError as exc:
        raise ImportError(
            "Weighted training requires torch and transformers. Install with: "
            'python3 -m pip install -e ".[train]"'
        ) from exc

    class PosWeightedTrainer(Trainer):
        def __init__(self, *args, pos_weights: list[float], **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.logits,
                labels,
                pos_weight=self.pos_weights.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    return PosWeightedTrainer
