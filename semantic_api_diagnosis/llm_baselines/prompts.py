"""Prompt builders for zero-shot and few-shot LLM baselines."""

import json

from semantic_api_diagnosis.dataset_plan import PLANNED_UNSEEN_ENDPOINT_FAMILIES
from semantic_api_diagnosis.labels import FIXED_LABEL_TAXONOMY


LABELS = sorted(FIXED_LABEL_TAXONOMY)
VALIDITY_VALUES = ["valid", "invalid"]
SEVERITY_VALUES = ["none", "medium", "high"]
FINAL_JSON_ONLY_INSTRUCTION = (
    "Return only one valid JSON object.\n"
    "Do not include markdown.\n"
    "Do not include explanations.\n"
    "Do not include any text before or after the JSON."
)


def build_zero_shot_prompt(example: dict) -> str:
    return _task_header() + "\n\n" + _input_block(example) + "\n\n" + _final_json_instruction()


def build_few_shot_prompt(example: dict, train_examples: list[dict], max_examples: int = 5) -> str:
    demonstrations = select_few_shot_examples(train_examples, max_examples=max_examples)
    demo_blocks = []
    for index, demo in enumerate(demonstrations, start=1):
        demo_blocks.append(
            "Demonstration {index}\n"
            "{input_block}\n"
            "JSON output:\n"
            "{target_json}".format(
                index=index,
                input_block=_input_block(demo),
                target_json=json.dumps(_target_json(demo), sort_keys=True),
            )
        )
    return (
        _task_header()
        + "\n\n"
        + "\n\n".join(demo_blocks)
        + "\n\nNow diagnose this request. Use the same JSON schema.\n"
        + _input_block(example)
        + "\n\n"
        + _final_json_instruction()
    )


def select_few_shot_examples(train_examples: list[dict], max_examples: int = 5) -> list[dict]:
    seen_train = [
        example
        for example in train_examples
        if example.get("endpoint_family") not in PLANNED_UNSEEN_ENDPOINT_FAMILIES
    ]
    selectors = [
        lambda example: example["target"]["validity"] == "valid",
        lambda example: _has_structural_error(example),
        lambda example: "semantic_cross_field_violation" in example["target"]["error_labels"],
        lambda example: bool(
            {"semantic_domain_constraint_violation", "semantic_state_violation"}
            & set(example["target"]["error_labels"])
        ),
        lambda example: len(example["target"]["error_labels"]) > 1,
    ]
    selected = []
    seen_ids = set()
    for selector in selectors:
        for example in seen_train:
            if example["id"] not in seen_ids and selector(example):
                selected.append(example)
                seen_ids.add(example["id"])
                break
        if len(selected) >= max_examples:
            return selected
    for example in seen_train:
        if example["id"] not in seen_ids:
            selected.append(example)
            seen_ids.add(example["id"])
        if len(selected) >= max_examples:
            break
    return selected


def _task_header() -> str:
    return (
        "You are diagnosing API request errors against an endpoint contract.\n"
        "Return strict JSON only. Do not include markdown, prose, or extra text.\n"
        "Multiple error labels are possible.\n"
        "Predict exactly these fields: validity, error_labels, severity_bucket.\n"
        f"validity must be one of: {', '.join(VALIDITY_VALUES)}.\n"
        f"severity_bucket must be one of: {', '.join(SEVERITY_VALUES)}.\n"
        "Allowed error label taxonomy:\n"
        + "\n".join(f"- {label}" for label in LABELS)
        + "\n"
        'Output schema: {"validity":"valid|invalid","error_labels":["label"],"severity_bucket":"none|medium|high"}'
    )


def _input_block(example: dict) -> str:
    return "Serialized input:\n" + example["serialized_input"]


def _final_json_instruction() -> str:
    return FINAL_JSON_ONLY_INSTRUCTION


def _target_json(example: dict) -> dict:
    target = example["target"]
    return {
        "validity": target["validity"],
        "error_labels": target["error_labels"],
        "severity_bucket": target["severity_bucket"],
    }


def _has_structural_error(example: dict) -> bool:
    return any(not label.startswith("semantic_") for label in example["target"]["error_labels"])
