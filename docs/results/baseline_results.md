# Baseline Evaluation Results

## Interpretation

- Majority and family-frequency baselines are weak, as expected for a multi-label diagnosis task.
- The visible rule-based baseline is strong on full and request-only views, indicating many request-obvious examples.
- Contract-dependent subsets provide a harder evaluation target focused on examples that need endpoint policy or constraints.
- Future model results should be reported both on the full test sets and on the contract-dependent subsets.

## Full Split Results

| split | baseline | micro-F1 | macro-F1 | exact match | validity acc | severity acc |
|---|---:|---:|---:|---:|---:|---:|
| validation | majority_empty | 0.000 | 0.000 | 0.224 | 0.776 | 0.560 |
| validation | majority_frequent | 0.000 | 0.000 | 0.224 | 0.776 | 0.560 |
| validation | family_frequency | 0.000 | 0.000 | 0.224 | 0.776 | 0.560 |
| validation | visible_rule_based | 0.817 | 0.811 | 0.684 | 0.846 | 0.834 |
| seen_test | majority_empty | 0.000 | 0.000 | 0.229 | 0.771 | 0.577 |
| seen_test | majority_frequent | 0.000 | 0.000 | 0.229 | 0.771 | 0.577 |
| seen_test | family_frequency | 0.000 | 0.000 | 0.229 | 0.771 | 0.577 |
| seen_test | visible_rule_based | 0.804 | 0.802 | 0.657 | 0.836 | 0.820 |
| unseen_family_test | majority_empty | 0.000 | 0.000 | 0.274 | 0.726 | 0.531 |
| unseen_family_test | majority_frequent | 0.000 | 0.000 | 0.274 | 0.726 | 0.531 |
| unseen_family_test | family_frequency | 0.000 | 0.000 | 0.274 | 0.726 | 0.531 |
| unseen_family_test | visible_rule_based | 0.923 | 0.887 | 0.874 | 0.931 | 0.931 |

## Semantic Label Performance

| split | baseline | semantic micro-F1 | semantic macro-F1 | per semantic label F1 |
|---|---:|---:|---:|---|
| validation | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | visible_rule_based | 0.904 | 0.898 | semantic_cross_field_violation=0.831, semantic_domain_constraint_violation=0.864, semantic_state_violation=1.000 |
| seen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | visible_rule_based | 0.872 | 0.866 | semantic_cross_field_violation=0.757, semantic_domain_constraint_violation=0.840, semantic_state_violation=1.000 |
| unseen_family_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | visible_rule_based | 0.960 | 0.957 | semantic_cross_field_violation=1.000, semantic_domain_constraint_violation=0.872, semantic_state_violation=1.000 |

## Critical Semantic Miss Rate

| split | baseline | semantic miss rate | high-severity semantic miss rate |
|---|---:|---:|---:|
| validation | majority_empty | 1.000 | 1.000 |
| validation | majority_frequent | 1.000 | 1.000 |
| validation | family_frequency | 1.000 | 1.000 |
| validation | visible_rule_based | 0.175 | 0.175 |
| seen_test | majority_empty | 1.000 | 1.000 |
| seen_test | majority_frequent | 1.000 | 1.000 |
| seen_test | family_frequency | 1.000 | 1.000 |
| seen_test | visible_rule_based | 0.228 | 0.228 |
| unseen_family_test | majority_empty | 1.000 | 1.000 |
| unseen_family_test | majority_frequent | 1.000 | 1.000 |
| unseen_family_test | family_frequency | 1.000 | 1.000 |
| unseen_family_test | visible_rule_based | 0.076 | 0.076 |

## Hard Contract-Dependent Subset Results

| split | baseline | micro-F1 | macro-F1 | exact match | validity acc | severity acc |
|---|---:|---:|---:|---:|---:|---:|
| contract_dependent_seen_test | majority_empty | 0.000 | 0.000 | 0.000 | 1.000 | 0.966 |
| contract_dependent_seen_test | majority_frequent | 0.000 | 0.000 | 0.000 | 1.000 | 0.966 |
| contract_dependent_seen_test | family_frequency | 0.000 | 0.000 | 0.000 | 1.000 | 0.966 |
| contract_dependent_seen_test | visible_rule_based | 0.588 | 0.213 | 0.343 | 0.600 | 0.554 |
| contract_dependent_unseen_test | majority_empty | 0.000 | 0.000 | 0.000 | 1.000 | 0.876 |
| contract_dependent_unseen_test | majority_frequent | 0.000 | 0.000 | 0.000 | 1.000 | 0.876 |
| contract_dependent_unseen_test | family_frequency | 0.000 | 0.000 | 0.000 | 1.000 | 0.876 |
| contract_dependent_unseen_test | visible_rule_based | 0.751 | 0.287 | 0.559 | 0.702 | 0.702 |

### Hard Subset Semantic Performance

| split | baseline | semantic micro-F1 | semantic macro-F1 | per semantic label F1 |
|---|---:|---:|---:|---|
| contract_dependent_seen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | visible_rule_based | 0.835 | 0.607 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.821, semantic_state_violation=1.000 |
| contract_dependent_unseen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | visible_rule_based | 0.939 | 0.623 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.870, semantic_state_violation=1.000 |

### Hard Subset Critical Semantic Miss Rate

| split | baseline | semantic miss rate | high-severity semantic miss rate |
|---|---:|---:|---:|
| contract_dependent_seen_test | majority_empty | 1.000 | 1.000 |
| contract_dependent_seen_test | majority_frequent | 1.000 | 1.000 |
| contract_dependent_seen_test | family_frequency | 1.000 | 1.000 |
| contract_dependent_seen_test | visible_rule_based | 0.283 | 0.283 |
| contract_dependent_unseen_test | majority_empty | 1.000 | 1.000 |
| contract_dependent_unseen_test | majority_frequent | 1.000 | 1.000 |
| contract_dependent_unseen_test | family_frequency | 1.000 | 1.000 |
| contract_dependent_unseen_test | visible_rule_based | 0.114 | 0.114 |

## Shortcut Comparison

| evaluation | baseline | micro-F1 | semantic macro-F1 | exact match | validity acc |
|---|---:|---:|---:|---:|---:|
| seen_test_full | visible_rule_based | 0.804 | 0.866 | 0.657 | 0.836 |
| unseen_family_test_full | visible_rule_based | 0.923 | 0.957 | 0.874 | 0.931 |
| seen_test_request_only | family_frequency | 0.000 | 0.000 | 0.229 | 0.771 |
| seen_test_request_only | visible_rule_based | 0.607 | 0.306 | 0.380 | 0.737 |
| seen_test_no_constraints | family_frequency | 0.000 | 0.000 | 0.229 | 0.771 |
| seen_test_no_constraints | visible_rule_based | 0.607 | 0.306 | 0.380 | 0.737 |
| seen_test_family_only | family_frequency | 0.000 | 0.000 | 0.229 | 0.771 |
| seen_test_family_only | visible_rule_based | 0.000 | 0.000 | 0.229 | 0.229 |
| unseen_family_test_request_only | family_frequency | 0.000 | 0.000 | 0.274 | 0.726 |
| unseen_family_test_request_only | visible_rule_based | 0.737 | 0.625 | 0.549 | 0.797 |
| unseen_family_test_no_constraints | family_frequency | 0.000 | 0.000 | 0.274 | 0.726 |
| unseen_family_test_no_constraints | visible_rule_based | 0.737 | 0.625 | 0.549 | 0.797 |
| unseen_family_test_family_only | family_frequency | 0.000 | 0.000 | 0.274 | 0.726 |
| unseen_family_test_family_only | visible_rule_based | 0.000 | 0.000 | 0.274 | 0.274 |

## Contract-Dependence Summary

### Bucket Distribution

| split | valid | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|---:|
| seen_test | 160 (22.9%) | 309 (44.1%) | 175 (25.0%) | 56 (8.0%) |
| unseen_family_test | 192 (27.4%) | 291 (41.6%) | 161 (23.0%) | 56 (8.0%) |

### Semantic Examples by Bucket

| split | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| seen_test | 42 (20.8%) | 127 (62.9%) | 33 (16.3%) |
| unseen_family_test | 66 (33.5%) | 105 (53.3%) | 26 (13.2%) |

### Visible Rule-Based on Contract-Dependent Subsets

| subset | semantic macro-F1 | semantic micro-F1 | critical semantic miss rate |
|---|---:|---:|---:|
| contract_dependent_seen_test | 0.607 | 0.835 | 0.283 |
| contract_dependent_unseen_test | 0.623 | 0.939 | 0.114 |
