"""Per-example visible contract policies for counterfactual generation."""

from __future__ import annotations

from copy import deepcopy
from random import Random


STATE_POLICIES = {
    "refunds_orders": ("order_status", ("paid", "pending")),
    "booking_reservation": ("reservation_status", ("pending", "confirmed", "checked_in")),
    "user_permissions": ("request_status", ("pending", "approved")),
    "inventory_product_search": ("reservation_status", ("available", "pending", "locked")),
    "payments_invoices": ("invoice_status", ("open", "partially_paid", "closed")),
    "healthcare_appointments": ("appointment_status", ("draft", "requested", "booked")),
    "course_registration": ("enrollment_status", ("eligible", "waitlisted", "graduated")),
    "ticket_support_workflow": ("current_status", ("new", "triaged", "in_progress")),
    "shipping_returns": ("delivery_status", ("delivered", "in_transit", "returned")),
    "subscription_plan_changes": ("account_status", ("active", "paused")),
}

MINIMUM_POLICIES = {
    "refunds_orders": ("refund_amount", (5.0, 10.0, 20.0)),
    "booking_reservation": ("number_of_guests", (2, 3)),
    "inventory_product_search": ("requested_quantity", (2, 3, 4)),
    "payments_invoices": ("payment_amount", (5.0, 10.0, 20.0)),
    "course_registration": ("requested_course_credits", (2, 3)),
    "shipping_returns": ("days_since_delivery", (2, 3)),
}

ALLOWED_VALUE_POLICIES = {
    "healthcare_appointments": (
        "appointment_type",
        (("primary_care", "physical_therapy"), ("primary_care", "covered_specialist")),
    ),
    "ticket_support_workflow": (
        "priority",
        (("low", "normal"), ("normal", "high")),
    ),
    "subscription_plan_changes": (
        "target_plan",
        (("pro",), ("enterprise",)),
    ),
}

MAXIMUM_POLICIES = {
    "refunds_orders": ("refund_amount", (40.0, 80.0, 120.0)),
    "booking_reservation": ("number_of_guests", (3, 4, 5)),
    "inventory_product_search": ("requested_quantity", (5, 10, 15)),
    "payments_invoices": ("payment_amount", (50.0, 100.0, 200.0)),
    "shipping_returns": ("days_since_delivery", (7, 14, 30)),
}


def add_counterfactual_policy(contract: dict, family: str, rng: Random) -> dict:
    """Attach a visible, randomized policy while retaining the base contract."""
    varied = deepcopy(contract)
    policy: dict[str, dict] = {}
    extra_constraints = []

    if family in STATE_POLICIES:
        field, values = STATE_POLICIES[family]
        allowed = (rng.choice(values),)
        policy["state"] = {"field": field, "allowed": list(allowed), "candidates": list(values)}
        varied["constraints"] = [
            constraint
            for constraint in varied["constraints"]
            if not (
                constraint.get("constraint_type") == "semantic_state"
                and field in constraint.get("fields", [])
            )
        ]
        extra_constraints.append(_allowed_constraint(field, allowed))

    if family in MINIMUM_POLICIES:
        field, values = MINIMUM_POLICIES[family]
        minimum = rng.choice(values)
        policy["minimum"] = {"field": field, "value": minimum}
        extra_constraints.append(_minimum_constraint(field, minimum))

    if family in ALLOWED_VALUE_POLICIES:
        field, choices = ALLOWED_VALUE_POLICIES[family]
        allowed = rng.choice(choices)
        candidates = sorted({value for choice in choices for value in choice})
        policy["allowed_values"] = {"field": field, "allowed": list(allowed), "candidates": candidates}
        extra_constraints.append(_allowed_constraint(field, allowed))

    if family in MAXIMUM_POLICIES:
        field, values = MAXIMUM_POLICIES[family]
        maximum = rng.choice(values)
        policy["maximum"] = {"field": field, "value": maximum}
        extra_constraints.append(_maximum_constraint(field, maximum))

    varied["policy"] = policy
    varied["constraints"] = varied["constraints"] + extra_constraints
    return varied


def apply_policy_to_valid_body(body: dict, policy: dict, rng: Random) -> None:
    state = policy.get("state")
    if state:
        body[state["field"]] = rng.choice(state["allowed"])

    allowed_values = policy.get("allowed_values")
    if allowed_values:
        body[allowed_values["field"]] = rng.choice(allowed_values["allowed"])

    minimum = policy.get("minimum")
    maximum = policy.get("maximum")
    if minimum and maximum and minimum["field"] == maximum["field"]:
        body[minimum["field"]] = _bounded_value(minimum["value"], maximum["value"], rng)
    elif minimum:
        body[minimum["field"]] = minimum["value"]
    elif maximum:
        body[maximum["field"]] = maximum["value"]

    _keep_related_values_valid(body, policy)


def inject_policy_error(example: dict, label: str, rng: Random) -> bool:
    body = example.get("request", {}).get("body")
    policy = example.get("endpoint_contract", {}).get("policy", {})
    if not isinstance(body, dict):
        return False

    if label == "semantic_state_violation" and policy.get("state"):
        rule = policy["state"]
        alternatives = [value for value in rule["candidates"] if value not in rule["allowed"]]
        if rule["field"] == "current_status":
            alternatives = [value for value in alternatives if value != body.get("target_status")]
        body[rule["field"]] = rng.choice(alternatives)
        return True
    if label == "invalid_value_range" and policy.get("allowed_values"):
        rule = policy["allowed_values"]
        alternatives = [value for value in rule["candidates"] if value not in rule["allowed"]]
        body[rule["field"]] = rng.choice(alternatives)
        return True
    if label == "invalid_value_range" and policy.get("minimum"):
        rule = policy["minimum"]
        body[rule["field"]] = _below(rule["value"])
        return True
    if label == "semantic_domain_constraint_violation" and policy.get("maximum"):
        rule = policy["maximum"]
        body[rule["field"]] = _above(rule["value"])
        _keep_related_values_valid(body, policy, injected=True)
        return True
    return False


def policy_findings(example: dict) -> list[tuple[str, str]]:
    body = example.get("request", {}).get("body")
    policy = example.get("endpoint_contract", {}).get("policy", {})
    if not isinstance(body, dict):
        return []
    findings = []

    state = policy.get("state")
    if state and state["field"] in body and body[state["field"]] not in state["allowed"]:
        findings.append(("semantic_state_violation", state["field"]))

    allowed_values = policy.get("allowed_values")
    if (
        allowed_values
        and allowed_values["field"] in body
        and body[allowed_values["field"]] not in allowed_values["allowed"]
    ):
        findings.append(("invalid_value_range", allowed_values["field"]))

    minimum = policy.get("minimum")
    if minimum and _number(body.get(minimum["field"])) is not None:
        if body[minimum["field"]] < minimum["value"]:
            findings.append(("invalid_value_range", minimum["field"]))

    maximum = policy.get("maximum")
    if maximum and _number(body.get(maximum["field"])) is not None:
        if body[maximum["field"]] > maximum["value"]:
            findings.append(("semantic_domain_constraint_violation", maximum["field"]))
    return findings


def _allowed_constraint(field: str, allowed: tuple | list) -> dict:
    values = ", ".join(str(value) for value in allowed)
    return _constraint(f"For this endpoint, {field} must be one of: {values}.", [field])


def _minimum_constraint(field: str, value: int | float) -> dict:
    return _constraint(f"For this endpoint, {field} must be at least {value}.", [field])


def _maximum_constraint(field: str, value: int | float) -> dict:
    return _constraint(f"For this endpoint, {field} must not exceed {value}.", [field])


def _constraint(text: str, fields: list[str]) -> dict:
    return {
        "constraint_id": "counterfactual_policy",
        "constraint_text": text,
        "constraint_type": "contract_local",
        "fields": fields,
        "severity": "high",
    }


def _bounded_value(minimum: int | float, maximum: int | float, rng: Random):
    if isinstance(minimum, float) or isinstance(maximum, float):
        return round(rng.uniform(float(minimum), float(maximum)), 2)
    return rng.randint(int(minimum), int(maximum))


def _below(value: int | float):
    return round(value - 1.0, 2) if isinstance(value, float) else value - 1


def _above(value: int | float):
    return round(value + 1.0, 2) if isinstance(value, float) else value + 1


def _keep_related_values_valid(body: dict, policy: dict, injected: bool = False) -> None:
    state = policy.get("state")
    if state and state["field"] == "current_status" and body.get("current_status") == body.get("target_status"):
        body["target_status"] = "resolved" if body["current_status"] != "resolved" else "in_progress"

    maximum = policy.get("maximum")
    if not maximum:
        return
    field = maximum["field"]
    value = body.get(field)
    if field == "refund_amount" and isinstance(value, (int, float)):
        body["original_payment"] = max(body.get("original_payment", 0), value + 25.0)
    elif field == "number_of_guests" and isinstance(value, int):
        body["room_capacity"] = max(body.get("room_capacity", 0), value + 1)
    elif field == "requested_quantity" and isinstance(value, int):
        body["available_stock"] = max(body.get("available_stock", 0), value + 3)
        body["allow_backorder"] = False
    elif field == "payment_amount" and isinstance(value, (int, float)):
        body["amount_already_paid"] = 0.0
        body["invoice_total"] = max(body.get("invoice_total", 0), value + 50.0)
    elif field == "days_since_delivery" and isinstance(value, int):
        body["return_window_days"] = max(body.get("return_window_days", 0), value + 5)

def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
