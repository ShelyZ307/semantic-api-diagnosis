from copy import deepcopy

from semantic_api_diagnosis.baselines.visible_rule_based import VisibleRuleBasedBaseline
from semantic_api_diagnosis.data_generation.generate import _build_valid_example, generate_examples
from semantic_api_diagnosis.evaluation.metrics import evaluate_predictions
from semantic_api_diagnosis.hidden_validators.validator import labels_for
from semantic_api_diagnosis.quality_gate import _request_only_input
from semantic_api_diagnosis.serialization.input_text import serialize_model_input


def test_same_request_changes_label_when_visible_contract_policy_changes() -> None:
    valid = _build_valid_example(1, "refunds_orders", _rng(), "train")
    counterfactual = deepcopy(valid)
    status = valid["request"]["body"]["order_status"]
    alternative = "pending" if status == "paid" else "paid"
    counterfactual["endpoint_contract"]["policy"]["state"]["allowed"] = [alternative]
    counterfactual["endpoint_contract"]["constraints"][-3]["constraint_text"] = (
        f"For this endpoint, order_status must be one of: {alternative}."
    )
    counterfactual["serialized_input"] = serialize_model_input(
        counterfactual["endpoint_contract"],
        counterfactual["request"],
    )

    assert labels_for(valid) == []
    assert "semantic_state_violation" in labels_for(counterfactual)
    assert valid["request"] == counterfactual["request"]
    assert valid["serialized_input"] != counterfactual["serialized_input"]


def test_semantic_identifier_errors_do_not_use_mismatch_marker() -> None:
    rows = generate_examples(500, split="train", family_mode="seen", seed=401)

    assert all("_mismatch" not in row["serialized_input"] for row in rows)


def test_visible_contract_rules_outperform_request_only_rules_after_tightening() -> None:
    rows = generate_examples(500, split="seen_test", family_mode="seen", seed=402)
    request_only = []
    for row in rows:
        view = deepcopy(row)
        view["serialized_input"] = _request_only_input(row)
        request_only.append(view)
    baseline = VisibleRuleBasedBaseline().fit([])
    full = evaluate_predictions(rows, baseline.predict(rows))
    shortcut = evaluate_predictions(request_only, baseline.predict(request_only))

    assert full["semantic_labels"]["macro_f1"] > shortcut["semantic_labels"]["macro_f1"] + 0.20
    assert full["critical_semantic_error_miss_rate"] < shortcut["critical_semantic_error_miss_rate"]


def test_counterfactual_clauses_appear_in_seen_and_unseen_generation() -> None:
    seen = generate_examples(100, split="train", family_mode="seen", seed=403)
    unseen = generate_examples(100, split="unseen_family_test", family_mode="unseen", seed=404)

    assert all("For this endpoint" in row["serialized_input"] for row in seen + unseen)


def _rng():
    from random import Random

    return Random(42)
