# Final Project Results Summary

## Dataset Summary

Version 1 uses a controlled synthetic dataset for semantic API request diagnosis from natural-language endpoint contracts.

Final local splits:

| split | examples | endpoint families |
|---|---:|---|
| train | 3000 | seen families only |
| validation | 500 | seen families only |
| seen_test | 700 | seen families only |
| unseen_family_test | 700 | held-out unseen families only |

Generated datasets are intentionally ignored by git and can be regenerated from the scripts.

Contract-dependent hard subsets:

| subset | examples |
|---|---:|
| contract_dependent_seen_test | 175 |
| contract_dependent_unseen_test | 161 |

## Endpoint Families

Seen families:

- `refunds_orders`
- `booking_reservation`
- `user_permissions`
- `inventory_product_search`
- `payments_invoices`

Held-out unseen families:

- `healthcare_appointments`
- `course_registration`
- `ticket_support_workflow`
- `shipping_returns`
- `subscription_plan_changes`

The code and generated final splits preserve the intended seen/unseen separation.

## Label Taxonomy

Structural / contract labels:

- `missing_required_field`
- `wrong_type`
- `invalid_value_range`
- `malformed_url`
- `wrong_http_method`
- `missing_authentication`
- `unexpected_or_malformed_body_structure`

Semantic labels:

- `semantic_cross_field_violation`
- `semantic_domain_constraint_violation`
- `semantic_state_violation`

## Baseline Results

The visible rule-based baseline is strong, especially on unseen-family semantic labels.

| view | model | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |
|---|---|---:|---:|---:|---:|
| seen_test | visible rule baseline | 0.804 | 0.866 | 0.657 | 0.228 |
| unseen_family_test | visible rule baseline | 0.923 | 0.957 | 0.874 | 0.076 |
| contract_dependent_seen_test | visible rule baseline | 0.588 | 0.607 | 0.343 | 0.283 |
| contract_dependent_unseen_test | visible rule baseline | 0.751 | 0.623 | 0.559 | 0.114 |

The majority and family-frequency baselines predict no useful error labels and have zero label F1, so the visible rule baseline is the meaningful non-neural comparison.

## Fine-Tuned Model Results

DistilBERT and RoBERTa were trained and evaluated with validation-tuned thresholds.

| view | model | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |
|---|---|---:|---:|---:|---:|
| seen_test | DistilBERT | 0.662 | 0.281 | 0.374 | 0.564 |
| seen_test | RoBERTa | 0.949 | 0.807 | 0.903 | 0.272 |
| unseen_family_test | DistilBERT | 0.587 | 0.179 | 0.403 | 0.782 |
| unseen_family_test | RoBERTa | 0.721 | 0.352 | 0.606 | 0.558 |
| contract_dependent_seen_test | RoBERTa | 0.927 | 0.839 | 0.840 | 0.220 |
| contract_dependent_unseen_test | RoBERTa | 0.420 | 0.263 | 0.217 | 0.524 |

RoBERTa is the strongest neural model. It beats the visible rule baseline on some seen-family all-label metrics and on contract-dependent seen examples, but it does not beat the visible rule baseline on unseen-family semantic diagnosis.

## Weighted RoBERTa

A single positive-class-weighted RoBERTa run was tried as a controlled Stage 7 improvement attempt.

| view | original RoBERTa semantic macro-F1 | weighted RoBERTa semantic macro-F1 | original exact | weighted exact |
|---|---:|---:|---:|---:|
| seen_test | 0.807 | 0.196 | 0.903 | 0.064 |
| unseen_family_test | 0.352 | 0.192 | 0.606 | 0.053 |
| contract_dependent_unseen_test | 0.263 | 0.328 | 0.217 | 0.000 |

Weighted RoBERTa is rejected as an improvement. Its lower critical miss rate comes from broad semantic overprediction, not better diagnosis.

## Shortcut And Contract-Dependence Interpretation

RoBERTa shows real contract sensitivity:

| split | full semantic macro-F1 | request-only | no constraints |
|---|---:|---:|---:|
| seen_test | 0.807 | 0.200 | 0.184 |
| unseen_family_test | 0.352 | 0.123 | 0.129 |

This supports the claim that RoBERTa uses contract text. However, the absolute unseen-family semantic score remains weak.

The visible rule baseline also remains strong. This means many examples are still partially request-obvious or rule-detectable, and the most scientifically meaningful analysis should focus on contract-dependent subsets and shortcut ablations.

## LLM Baseline Status

The Stage 8 fixed sample protocol exists:

- 40 seen examples
- 40 unseen-family examples
- 40 contract-dependent unseen examples
- seed `808`

Same-sample non-LLM results are available in `docs/results/stage_8_llm_sample_baselines.md`.

Provider-backed LLM results are not available because `OPENAI_API_KEY` was unavailable. Mock LLM outputs must not be cited as scientific evidence.

## Final Scientific Interpretation

The strongest supported story is:

> This project builds a controlled benchmark for diagnosing semantic API request errors from natural-language endpoint contracts. It shows that many generated examples are request-obvious and that transparent visible-rule validation is a strong baseline. Fine-tuned RoBERTa learns contract-sensitive behavior and performs strongly in-domain, but it struggles to generalize semantic reasoning to held-out endpoint families. The main contribution is the dataset, evaluation protocol, hard subsets, shortcut views, and analysis of where contract-dependent semantic diagnosis remains difficult.

Risky claim to avoid:

> RoBERTa beats all baselines or solves unseen-family semantic API diagnosis.

That claim is not supported by the current results.
