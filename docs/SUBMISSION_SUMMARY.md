# Submission Summary

## Project title

**Semantic API Request Diagnosis from Natural-Language Endpoint Contracts**

## One-sentence summary

This project builds and evaluates a controlled NLP benchmark for diagnosing structural and semantic API request errors from natural-language endpoint contracts, with special focus on contract-dependent cases and held-out unseen endpoint-family generalization.

---

## Motivation

Backend developers and QA testers often need to understand why an API request is invalid. Ordinary schema validators can catch missing fields, wrong types, malformed URLs, wrong HTTP methods, and authentication problems. The harder cases are semantic: a request may be structurally valid but violate a natural-language contract rule.

Example:

```text
Contract rule:
The requested refund must not be greater than the original payment.

Request:
refund_amount: 120
original_payment: 80

Expected diagnosis:
semantic_domain_constraint_violation
```

The project asks whether language models can use natural-language endpoint constraints to diagnose those errors, and whether that behavior generalizes beyond the endpoint families seen during training.

---

## Task definition

Each example contains:

- endpoint description
- natural-language constraints
- serialized API request
- target validity
- target error-label set
- target severity bucket

Primary task:

- multi-label classification over API request error labels

Secondary tasks:

- binary validity prediction
- severity-bucket prediction

The fixed Version 1 taxonomy contains 10 labels:

| Structural / contract labels | Semantic labels |
|---|---|
| `missing_required_field` | `semantic_cross_field_violation` |
| `wrong_type` | `semantic_domain_constraint_violation` |
| `invalid_value_range` | `semantic_state_violation` |
| `malformed_url` | |
| `wrong_http_method` | |
| `missing_authentication` | |
| `unexpected_or_malformed_body_structure` | |

---

## Dataset

The Version 1 dataset is generated through controlled synthetic data generation:

```text
valid request -> controlled error injection -> hidden validation -> deterministic labels -> text serialization
```

The model does not receive hidden validator logic. Hidden validators are used only to generate and audit deterministic labels.

Final generated splits:

| Split | Size | Families |
|---|---:|---|
| `final_train.jsonl` | 3000 | seen endpoint families only |
| `final_validation.jsonl` | 500 | seen endpoint families only |
| `final_seen_test.jsonl` | 700 | seen endpoint families only |
| `final_unseen_family_test.jsonl` | 700 | held-out unseen endpoint families only |
| **Total** | **4900** | seen + held-out unseen |

Seen endpoint families:

- `refunds_orders`
- `booking_reservation`
- `user_permissions`
- `inventory_product_search`
- `payments_invoices`

Held-out unseen endpoint families:

- `healthcare_appointments`
- `course_registration`
- `ticket_support_workflow`
- `shipping_returns`
- `subscription_plan_changes`

Quality gates passed:

- all labels appear in every split
- all semantic labels appear in every split
- no exact duplicate serialized inputs across splits
- no exact duplicate contract-plus-request pairs across splits
- no near-duplicate serialized inputs across splits above the configured threshold
- no seen/unseen family leakage
- hidden validator exposure checks passed

---

## Baselines and models

Implemented baseline/model families:

- `majority_empty`
- `majority_frequent`
- `family_frequency`
- `visible_rule_based`
- DistilBERT fine-tuning
- RoBERTa fine-tuning
- fixed-sample zero-shot/few-shot LLM baseline protocol

Large LLMs are used as zero/few-shot baselines only. Local encoder fine-tuning is used for the trained classification models.

---

## Evaluation design

The project reports performance across multiple views instead of relying on one random test split:

1. full seen-family test set
2. full held-out unseen-family test set
3. contract-dependent seen hard subset
4. contract-dependent unseen hard subset
5. shortcut-ablation views such as request-only and no-constraints

Main metrics:

- all-label micro-F1
- all-label macro-F1
- semantic-only micro-F1
- semantic-only macro-F1
- exact-match label-set accuracy
- validity accuracy
- severity accuracy
- critical semantic-error miss rate

The most important scientific check is whether performance drops when natural-language constraint text is removed. If a model performs similarly without constraints, it may be exploiting request-only shortcuts instead of using the endpoint contract.

---

## Main Version 1 findings

The honest conclusion is intentionally narrow:

- Majority and family-frequency baselines are weak, so simple label priors are not enough.
- The visible rule-based baseline is strong on full test sets, showing that many examples are request-obvious.
- Contract-dependent hard subsets are more scientifically useful because they focus on examples requiring endpoint policy or constraint information.
- Shortcut ablations now show a substantial semantic-performance decrease when contract constraints are removed.
- RoBERTa learns contract-sensitive behavior and performs strongly on seen endpoint families.
- RoBERTa does not solve unseen-family semantic generalization.
- RoBERTa does not beat the visible rule-based baseline on unseen semantic diagnosis.
- Provider-backed Stage 8 `gpt-4o-mini` zero-shot and few-shot calls completed on the fixed 120-example sample; they are treated as sampled baselines, not full-test LLM evidence.

Final claim:

> The project contributes a reproducible benchmark, controlled data-generation pipeline, leakage/shortcut quality gates, baseline suite, encoder training pipeline, contract-dependent evaluation protocol, and error-analysis framework for semantic API request diagnosis. It does not claim that fine-tuned encoders fully solve held-out endpoint-family reasoning.

---

## What this project is not claiming

This is not a project about:

- OpenAPI schema validation
- malformed JSON detection as a standalone task
- real API execution
- API gateway security
- UI/product deployment
- proprietary LLM fine-tuning

The project is about structured semantic diagnosis from natural-language endpoint contracts under controlled evaluation.

---

## Recommended reviewer path

1. Read `README.md` for the complete project overview and reproduction commands.
2. Read `docs/results/baseline_results.md` for baseline metrics and hard-subset analysis.
3. Read `docs/results/contract_dependence_tightening_report.md` for the shortcut-ablation tightening.
4. Review `scripts/` and `tests/` to see how generation, leakage checks, baseline evaluation, and model evaluation are implemented.
