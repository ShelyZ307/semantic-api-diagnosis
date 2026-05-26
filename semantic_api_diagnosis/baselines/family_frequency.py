"""Endpoint-family frequency baseline."""

from collections import Counter, defaultdict

from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY


class FamilyFrequencyBaseline:
    def __init__(self, label_threshold: float = 0.3) -> None:
        self.label_threshold = label_threshold
        self.global_labels: list[str] = []
        self.global_validity = "valid"
        self.global_severity = "none"
        self.family_labels: dict[str, list[str]] = {}
        self.family_validity: dict[str, str] = {}
        self.family_severity: dict[str, str] = {}

    def fit(self, examples: list[dict]) -> "FamilyFrequencyBaseline":
        self.global_labels = _frequent_labels(examples, self.label_threshold)
        self.global_validity = _majority(examples, "validity", "valid")
        self.global_severity = _majority(examples, "severity_bucket", "none")
        by_family = defaultdict(list)
        for example in examples:
            by_family[example["endpoint_family"]].append(example)
        for family, family_examples in by_family.items():
            self.family_labels[family] = _frequent_labels(family_examples, self.label_threshold)
            self.family_validity[family] = _majority(family_examples, "validity", self.global_validity)
            self.family_severity[family] = _majority(family_examples, "severity_bucket", self.global_severity)
        return self

    def predict_one(self, example: dict) -> dict:
        family = example.get("endpoint_family")
        labels = self.family_labels.get(family, self.global_labels)
        return {
            "error_labels": list(labels),
            "validity": self.family_validity.get(family, self.global_validity),
            "severity_bucket": self.family_severity.get(family, self.global_severity),
        }

    def predict(self, examples: list[dict]) -> list[dict]:
        return [self.predict_one(example) for example in examples]


def _frequent_labels(examples: list[dict], threshold: float) -> list[str]:
    total = len(examples) or 1
    counts = Counter(label for example in examples for label in example["target"]["error_labels"])
    return [
        label
        for label in sorted(FIXED_LABEL_TAXONOMY)
        if counts[label] / total >= threshold
    ]


def _majority(examples: list[dict], target_field: str, default: str) -> str:
    counts = Counter(example["target"][target_field] for example in examples)
    return counts.most_common(1)[0][0] if counts else default
