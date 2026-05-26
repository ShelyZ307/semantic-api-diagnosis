# Contract Dependence Analysis

## Bucket Distribution

| split | valid | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|---:|
| seen_test | 165 (23.6%) | 307 (43.9%) | 169 (24.1%) | 59 (8.4%) |
| unseen_family_test | 194 (27.7%) | 288 (41.1%) | 161 (23.0%) | 57 (8.1%) |

## Error Labels by Bucket

### seen_test

| label | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| invalid_value_range | 50 | 8 | 6 |
| malformed_url | 59 | 0 | 17 |
| missing_authentication | 66 | 0 | 6 |
| missing_required_field | 0 | 68 | 16 |
| semantic_cross_field_violation | 35 | 14 | 8 |
| semantic_domain_constraint_violation | 0 | 52 | 15 |
| semantic_state_violation | 0 | 54 | 14 |
| unexpected_or_malformed_body_structure | 46 | 0 | 0 |
| wrong_http_method | 45 | 0 | 31 |
| wrong_type | 57 | 0 | 5 |

### unseen_family_test

| label | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| invalid_value_range | 26 | 23 | 15 |
| malformed_url | 55 | 0 | 23 |
| missing_authentication | 58 | 0 | 6 |
| missing_required_field | 0 | 65 | 14 |
| semantic_cross_field_violation | 66 | 0 | 0 |
| semantic_domain_constraint_violation | 0 | 51 | 14 |
| semantic_state_violation | 0 | 50 | 14 |
| unexpected_or_malformed_body_structure | 37 | 0 | 0 |
| wrong_http_method | 36 | 0 | 28 |
| wrong_type | 65 | 0 | 0 |

## Semantic Examples

| split | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| seen_test | 35 (18.2%) | 120 (62.5%) | 37 (19.3%) |
| unseen_family_test | 66 (33.8%) | 101 (51.8%) | 28 (14.4%) |

## Visible Rule-Based Performance by Bucket

| split | bucket | micro-F1 | semantic macro-F1 | critical semantic miss rate | exact match |
|---|---|---:|---:|---:|---:|
| seen_test | request_obvious | 0.983 | 0.333 | 0.000 | 0.961 |
| seen_test | contract_dependent | 0.680 | 0.619 | 0.225 | 0.462 |
| seen_test | mixed | 0.865 | 0.643 | 0.270 | 0.559 |
| seen_test | valid | 0.000 | 0.000 | n/a | 1.000 |
| unseen_family_test | request_obvious | 1.000 | 0.333 | 0.000 | 1.000 |
| unseen_family_test | contract_dependent | 0.701 | 0.575 | 0.218 | 0.491 |
| unseen_family_test | mixed | 0.893 | 0.533 | 0.286 | 0.614 |
| unseen_family_test | valid | 0.000 | 0.000 | n/a | 1.000 |

## Full vs Request-Only Visible Baseline

| split | view | micro-F1 | semantic macro-F1 | critical semantic miss rate |
|---|---|---:|---:|---:|
| seen_test | visible_rule_full | 0.888 | 0.878 | 0.193 |
| seen_test | visible_rule_request_only | 0.888 | 0.878 | 0.193 |
| unseen_family_test | visible_rule_full | 0.908 | 0.900 | 0.154 |
| unseen_family_test | visible_rule_request_only | 0.908 | 0.900 | 0.154 |

## Warnings

- seen_test: request-only semantic macro-F1 is within 0.05 of full-input performance
- unseen_family_test: request-only semantic macro-F1 is within 0.05 of full-input performance
