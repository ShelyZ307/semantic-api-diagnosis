# Stage 7 Unseen-Family Error Analysis

This report compares the original and positive-class-weighted RoBERTa checkpoints on `final_unseen_family_test.jsonl`. Thresholds were tuned on validation only.

## original RoBERTa

- Semantic false negatives: `123`.
- Semantic false positives: `87`.
- Threshold-near semantic misses: `49`.
- Far-below-threshold semantic misses: `74`.
- Examples exceeding `max_length=512`: `0`; semantic misses on those examples: `0`.
- Endpoint-local policy fields observed in semantic misses: `account_status, appointment_status, appointment_type, current_status, days_since_delivery, delivery_status, enrollment_status, priority, requested_course_credits, target_plan`.

### Errors By Semantic Label

| label | false negatives | false positives | threshold-near misses |
|---|---:|---:|---:|
| semantic_cross_field_violation | 41 | 23 | 24 |
| semantic_domain_constraint_violation | 64 | 10 | 16 |
| semantic_state_violation | 18 | 54 | 9 |

### Errors By Endpoint Family

| endpoint family | semantic false negatives | semantic false positives |
|---|---:|---:|
| course_registration | 17 | 23 |
| healthcare_appointments | 31 | 0 |
| shipping_returns | 27 | 42 |
| subscription_plan_changes | 21 | 14 |
| ticket_support_workflow | 27 | 8 |

### Error Context

| context | semantic false negatives |
|---|---:|
| complexity=multi_error | 46 |
| complexity=single_error | 77 |
| constraint_type=semantic_cross_field | 41 |
| constraint_type=semantic_domain_or_contract_local | 64 |
| constraint_type=semantic_state_or_contract_local | 18 |
| dependence=contract_dependent | 62 |
| dependence=mixed | 20 |
| dependence=request_obvious | 41 |
| severity=high | 123 |

### Sample Semantic False Negatives

- `ex_000002` (`healthcare_appointments`): `semantic_cross_field_violation` score `0.084` vs threshold `0.250`; gold `['semantic_cross_field_violation']`; predicted `[]`.
- `ex_000003` (`course_registration`): `semantic_domain_constraint_violation` score `0.013` vs threshold `0.250`; gold `['semantic_domain_constraint_violation']`; predicted `['missing_authentication']`.
- `ex_000016` (`healthcare_appointments`): `semantic_domain_constraint_violation` score `0.057` vs threshold `0.250`; gold `['semantic_domain_constraint_violation']`; predicted `[]`.
- `ex_000028` (`ticket_support_workflow`): `semantic_cross_field_violation` score `0.167` vs threshold `0.250`; gold `['semantic_cross_field_violation']`; predicted `[]`.
- `ex_000030` (`shipping_returns`): `semantic_domain_constraint_violation` score `0.073` vs threshold `0.250`; gold `['semantic_domain_constraint_violation']`; predicted `[]`.
- `ex_000042` (`shipping_returns`): `semantic_cross_field_violation` score `0.084` vs threshold `0.250`; gold `['semantic_cross_field_violation']`; predicted `['semantic_state_violation']`.
- `ex_000043` (`subscription_plan_changes`): `semantic_domain_constraint_violation` score `0.105` vs threshold `0.250`; gold `['semantic_domain_constraint_violation']`; predicted `[]`.
- `ex_000055` (`subscription_plan_changes`): `semantic_cross_field_violation` score `0.192` vs threshold `0.250`; gold `['semantic_cross_field_violation']`; predicted `[]`.

### Sample Semantic False Positives

- `ex_000022` (`subscription_plan_changes`): `semantic_domain_constraint_violation` score `0.263` vs threshold `0.250`; gold `['missing_required_field']`; predicted `['missing_required_field', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000022` (`subscription_plan_changes`): `semantic_state_violation` score `0.605` vs threshold `0.500`; gold `['missing_required_field']`; predicted `['missing_required_field', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000029` (`course_registration`): `semantic_cross_field_violation` score `0.262` vs threshold `0.250`; gold `[]`; predicted `['semantic_cross_field_violation']`.
- `ex_000032` (`course_registration`): `semantic_cross_field_violation` score `0.265` vs threshold `0.250`; gold `['wrong_type']`; predicted `['wrong_type', 'semantic_cross_field_violation']`.
- `ex_000035` (`ticket_support_workflow`): `semantic_state_violation` score `0.746` vs threshold `0.500`; gold `['missing_required_field']`; predicted `['semantic_state_violation']`.
- `ex_000042` (`shipping_returns`): `semantic_state_violation` score `0.866` vs threshold `0.500`; gold `['semantic_cross_field_violation']`; predicted `['semantic_state_violation']`.
- `ex_000049` (`course_registration`): `semantic_cross_field_violation` score `0.276` vs threshold `0.250`; gold `[]`; predicted `['semantic_cross_field_violation']`.
- `ex_000062` (`course_registration`): `semantic_cross_field_violation` score `0.305` vs threshold `0.250`; gold `['missing_required_field']`; predicted `['semantic_cross_field_violation']`.

## weighted-loss RoBERTa

- Semantic false negatives: `0`.
- Semantic false positives: `1662`.
- Threshold-near semantic misses: `0`.
- Far-below-threshold semantic misses: `0`.
- Examples exceeding `max_length=512`: `0`; semantic misses on those examples: `0`.
- Endpoint-local policy fields observed in semantic misses: `none`.

### Errors By Semantic Label

| label | false negatives | false positives | threshold-near misses |
|---|---:|---:|---:|
| semantic_cross_field_violation | 0 | 597 | 0 |
| semantic_domain_constraint_violation | 0 | 532 | 0 |
| semantic_state_violation | 0 | 533 | 0 |

### Errors By Endpoint Family

| endpoint family | semantic false negatives | semantic false positives |
|---|---:|---:|
| course_registration | 0 | 334 |
| healthcare_appointments | 0 | 333 |
| shipping_returns | 0 | 332 |
| subscription_plan_changes | 0 | 330 |
| ticket_support_workflow | 0 | 333 |

### Error Context

| context | semantic false negatives |
|---|---:|

### Sample Semantic False Negatives

- None.

### Sample Semantic False Positives

- `ex_000001` (`healthcare_appointments`): `semantic_cross_field_violation` score `0.439` vs threshold `0.400`; gold `[]`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000001` (`healthcare_appointments`): `semantic_domain_constraint_violation` score `0.487` vs threshold `0.450`; gold `[]`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000001` (`healthcare_appointments`): `semantic_state_violation` score `0.500` vs threshold `0.450`; gold `[]`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000002` (`healthcare_appointments`): `semantic_domain_constraint_violation` score `0.487` vs threshold `0.450`; gold `['semantic_cross_field_violation']`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000002` (`healthcare_appointments`): `semantic_state_violation` score `0.501` vs threshold `0.450`; gold `['semantic_cross_field_violation']`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000003` (`course_registration`): `semantic_cross_field_violation` score `0.440` vs threshold `0.400`; gold `['semantic_domain_constraint_violation']`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000003` (`course_registration`): `semantic_state_violation` score `0.512` vs threshold `0.450`; gold `['semantic_domain_constraint_violation']`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.
- `ex_000004` (`ticket_support_workflow`): `semantic_cross_field_violation` score `0.440` vs threshold `0.400`; gold `['semantic_state_violation']`; predicted `['invalid_value_range', 'malformed_url', 'wrong_http_method', 'missing_authentication', 'semantic_cross_field_violation', 'semantic_domain_constraint_violation', 'semantic_state_violation']`.

## Comparative Interpretation

- Critical semantic false negatives: original `123`, weighted `0`.
- Threshold-near semantic misses: original `49`, weighted `0`. A threshold-near miss has a score at least half of its selected validation threshold.
- Semantic false positives: original `87`, weighted `1662`.
- Remaining misses with scores far below threshold are more consistent with unseen-template transfer failure than threshold calibration alone.
- The endpoint-local policies use unseen field names, enum/status vocabularies, and domain wording. Those shifts remain a central generalization challenge.
- Truncation is not a plausible explanation: no unseen example exceeds `max_length=512`.
- Original RoBERTa shows semantic-label confusion: cross-field violations are sometimes predicted as state violations, while valid or non-semantic rows can receive cross-field or state predictions.
- Weighted loss is rejected as an improvement. It removes semantic false negatives by predicting violations broadly, increasing semantic false positives from `87` to `1662`.
- A later provider-backed Stage 8 LLM sample run completed on the fixed 120-example sample, but it remains a sampled baseline and does not change the current unseen-family conclusion.
