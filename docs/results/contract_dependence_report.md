# Contract Dependence Analysis

## Bucket Distribution

| split | valid | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|---:|
| seen_test | 160 (22.9%) | 309 (44.1%) | 175 (25.0%) | 56 (8.0%) |
| unseen_family_test | 192 (27.4%) | 291 (41.6%) | 161 (23.0%) | 56 (8.0%) |

## Error Labels by Bucket

### seen_test

| label | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| invalid_value_range | 49 | 6 | 9 |
| malformed_url | 64 | 0 | 16 |
| missing_authentication | 65 | 0 | 10 |
| missing_required_field | 0 | 73 | 14 |
| semantic_cross_field_violation | 42 | 19 | 8 |
| semantic_domain_constraint_violation | 0 | 56 | 13 |
| semantic_state_violation | 0 | 52 | 12 |
| unexpected_or_malformed_body_structure | 45 | 0 | 0 |
| wrong_http_method | 40 | 0 | 27 |
| wrong_type | 61 | 0 | 3 |

### unseen_family_test

| label | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| invalid_value_range | 26 | 20 | 16 |
| malformed_url | 56 | 0 | 21 |
| missing_authentication | 59 | 0 | 7 |
| missing_required_field | 0 | 63 | 14 |
| semantic_cross_field_violation | 66 | 0 | 0 |
| semantic_domain_constraint_violation | 0 | 52 | 14 |
| semantic_state_violation | 0 | 53 | 12 |
| unexpected_or_malformed_body_structure | 37 | 0 | 0 |
| wrong_http_method | 37 | 0 | 28 |
| wrong_type | 65 | 0 | 0 |

## Semantic Examples

| split | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| seen_test | 42 (20.8%) | 127 (62.9%) | 33 (16.3%) |
| unseen_family_test | 66 (33.5%) | 105 (53.3%) | 26 (13.2%) |

## Visible Rule-Based Performance by Bucket

| split | bucket | micro-F1 | semantic macro-F1 | critical semantic miss rate | exact match |
|---|---|---:|---:|---:|---:|
| seen_test | request_obvious | 0.928 | 0.333 | 0.000 | 0.822 |
| seen_test | contract_dependent | 0.588 | 0.607 | 0.283 | 0.343 |
| seen_test | mixed | 0.834 | 0.639 | 0.303 | 0.429 |
| seen_test | valid | 0.000 | 0.000 | n/a | 0.762 |
| unseen_family_test | request_obvious | 1.000 | 0.333 | 0.000 | 1.000 |
| unseen_family_test | contract_dependent | 0.751 | 0.623 | 0.114 | 0.559 |
| unseen_family_test | mixed | 0.918 | 0.627 | 0.115 | 0.696 |
| unseen_family_test | valid | 0.000 | 0.000 | n/a | 1.000 |

## Full vs Request-Only Visible Baseline

| split | view | micro-F1 | semantic macro-F1 | critical semantic miss rate |
|---|---|---:|---:|---:|
| seen_test | visible_rule_full | 0.804 | 0.866 | 0.228 |
| seen_test | visible_rule_request_only | 0.607 | 0.306 | 0.436 |
| unseen_family_test | visible_rule_full | 0.923 | 0.957 | 0.076 |
| unseen_family_test | visible_rule_request_only | 0.737 | 0.625 | 0.310 |

## Warnings

- none
