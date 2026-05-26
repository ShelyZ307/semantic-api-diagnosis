"""Majority and frequent-label baselines."""

from collections import Counter

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY


class MajorityBaseline:
    def __init__(self, label_threshold: float | None = None) -> None:
        self.label_threshold = label_threshold
        self.validity = "valid"
        self.severity_bucket = "none"
        self.labels: list[str] = []

    def fit(self, examples: list[dict]) -> "MajorityBaseline":
        total = len(examples) or 1
        validity_counts = Counter(example["target"]["validity"] for example in examples)
        severity_counts = Counter(example["target"]["severity_bucket"] for example in examples)
        label_counts = Counter(
            label
            for example in examples
            for label in example["target"]["error_labels"]
        )
        self.validity = _most_common(validity_counts, default="valid")
        self.severity_bucket = _most_common(severity_counts, default="none")
        if self.label_threshold is None:
            self.labels = []
        else:
            self.labels = [
                label
                for label in sorted(FIXED_LABEL_TAXONOMY)
                if label_counts[label] / total >= self.label_threshold
            ]
        return self

    def predict_one(self, example: dict) -> dict:
        return {
            "error_labels": list(self.labels),
            "validity": self.validity,
            "severity_bucket": self.severity_bucket,
        }

    def predict(self, examples: list[dict]) -> list[dict]:
        return [self.predict_one(example) for example in examples]


def _most_common(counter: Counter, default: str) -> str:
    return counter.most_common(1)[0][0] if counter else default
