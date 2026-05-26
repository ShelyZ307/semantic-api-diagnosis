# Baseline Evaluation Results

## Interpretation

- Majority and family-frequency baselines are weak, as expected for a multi-label diagnosis task.
- The visible rule-based baseline is strong on full and request-only views, indicating many request-obvious examples.
- Contract-dependent subsets provide a harder evaluation target focused on examples that need endpoint policy or constraints.
- Future model results should be reported both on the full test sets and on the contract-dependent subsets.

## Full Split Results

| split | baseline | micro-F1 | macro-F1 | exact match | validity acc | severity acc |
|---|---:|---:|---:|---:|---:|---:|
| validation | majority_empty | 0.000 | 0.000 | 0.256 | 0.744 | 0.554 |
| validation | majority_frequent | 0.000 | 0.000 | 0.256 | 0.744 | 0.554 |
| validation | family_frequency | 0.000 | 0.000 | 0.256 | 0.744 | 0.554 |
| validation | visible_rule_based | 0.900 | 0.859 | 0.836 | 0.902 | 0.900 |
| seen_test | majority_empty | 0.000 | 0.000 | 0.236 | 0.764 | 0.571 |
| seen_test | majority_frequent | 0.000 | 0.000 | 0.236 | 0.764 | 0.571 |
| seen_test | family_frequency | 0.000 | 0.000 | 0.236 | 0.764 | 0.571 |
| seen_test | visible_rule_based | 0.888 | 0.851 | 0.816 | 0.887 | 0.883 |
| unseen_family_test | majority_empty | 0.000 | 0.000 | 0.277 | 0.723 | 0.526 |
| unseen_family_test | majority_frequent | 0.000 | 0.000 | 0.277 | 0.723 | 0.526 |
| unseen_family_test | family_frequency | 0.000 | 0.000 | 0.277 | 0.723 | 0.526 |
| unseen_family_test | visible_rule_based | 0.908 | 0.870 | 0.851 | 0.916 | 0.916 |

## Semantic Label Performance

| split | baseline | semantic micro-F1 | semantic macro-F1 | per semantic label F1 |
|---|---:|---:|---:|---|
| validation | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| validation | visible_rule_based | 0.911 | 0.907 | semantic_cross_field_violation=0.828, semantic_domain_constraint_violation=0.895, semantic_state_violation=1.000 |
| seen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| seen_test | visible_rule_based | 0.893 | 0.878 | semantic_cross_field_violation=0.761, semantic_domain_constraint_violation=0.874, semantic_state_violation=1.000 |
| unseen_family_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| unseen_family_test | visible_rule_based | 0.917 | 0.900 | semantic_cross_field_violation=1.000, semantic_domain_constraint_violation=0.700, semantic_state_violation=1.000 |

## Critical Semantic Miss Rate

| split | baseline | semantic miss rate | high-severity semantic miss rate |
|---|---:|---:|---:|
| validation | majority_empty | 1.000 | 1.000 |
| validation | majority_frequent | 1.000 | 1.000 |
| validation | family_frequency | 1.000 | 1.000 |
| validation | visible_rule_based | 0.163 | 0.163 |
| seen_test | majority_empty | 1.000 | 1.000 |
| seen_test | majority_frequent | 1.000 | 1.000 |
| seen_test | family_frequency | 1.000 | 1.000 |
| seen_test | visible_rule_based | 0.193 | 0.193 |
| unseen_family_test | majority_empty | 1.000 | 1.000 |
| unseen_family_test | majority_frequent | 1.000 | 1.000 |
| unseen_family_test | family_frequency | 1.000 | 1.000 |
| unseen_family_test | visible_rule_based | 0.154 | 0.154 |

## Hard Contract-Dependent Subset Results

| split | baseline | micro-F1 | macro-F1 | exact match | validity acc | severity acc |
|---|---:|---:|---:|---:|---:|---:|
| contract_dependent_seen_test | majority_empty | 0.000 | 0.000 | 0.000 | 1.000 | 0.953 |
| contract_dependent_seen_test | majority_frequent | 0.000 | 0.000 | 0.000 | 1.000 | 0.953 |
| contract_dependent_seen_test | family_frequency | 0.000 | 0.000 | 0.000 | 1.000 | 0.953 |
| contract_dependent_seen_test | visible_rule_based | 0.680 | 0.286 | 0.462 | 0.598 | 0.598 |
| contract_dependent_unseen_test | majority_empty | 0.000 | 0.000 | 0.000 | 1.000 | 0.857 |
| contract_dependent_unseen_test | majority_frequent | 0.000 | 0.000 | 0.000 | 1.000 | 0.857 |
| contract_dependent_unseen_test | family_frequency | 0.000 | 0.000 | 0.000 | 1.000 | 0.857 |
| contract_dependent_unseen_test | visible_rule_based | 0.701 | 0.273 | 0.491 | 0.634 | 0.634 |

### Hard Subset Semantic Performance

| split | baseline | semantic micro-F1 | semantic macro-F1 | per semantic label F1 |
|---|---:|---:|---:|---|
| contract_dependent_seen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_seen_test | visible_rule_based | 0.873 | 0.619 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.857, semantic_state_violation=1.000 |
| contract_dependent_unseen_test | majority_empty | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | majority_frequent | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | family_frequency | 0.000 | 0.000 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.000, semantic_state_violation=0.000 |
| contract_dependent_unseen_test | visible_rule_based | 0.878 | 0.575 | semantic_cross_field_violation=0.000, semantic_domain_constraint_violation=0.725, semantic_state_violation=1.000 |

### Hard Subset Critical Semantic Miss Rate

| split | baseline | semantic miss rate | high-severity semantic miss rate |
|---|---:|---:|---:|
| contract_dependent_seen_test | majority_empty | 1.000 | 1.000 |
| contract_dependent_seen_test | majority_frequent | 1.000 | 1.000 |
| contract_dependent_seen_test | family_frequency | 1.000 | 1.000 |
| contract_dependent_seen_test | visible_rule_based | 0.225 | 0.225 |
| contract_dependent_unseen_test | majority_empty | 1.000 | 1.000 |
| contract_dependent_unseen_test | majority_frequent | 1.000 | 1.000 |
| contract_dependent_unseen_test | family_frequency | 1.000 | 1.000 |
| contract_dependent_unseen_test | visible_rule_based | 0.218 | 0.218 |

## Shortcut Comparison

| evaluation | baseline | micro-F1 | semantic macro-F1 | exact match | validity acc |
|---|---:|---:|---:|---:|---:|
| seen_test_full | visible_rule_based | 0.888 | 0.878 | 0.816 | 0.887 |
| unseen_family_test_full | visible_rule_based | 0.908 | 0.900 | 0.851 | 0.916 |
| seen_test_request_only | family_frequency | 0.000 | 0.000 | 0.236 | 0.764 |
| seen_test_request_only | visible_rule_based | 0.888 | 0.878 | 0.816 | 0.887 |
| seen_test_no_constraints | family_frequency | 0.000 | 0.000 | 0.236 | 0.764 |
| seen_test_no_constraints | visible_rule_based | 0.888 | 0.878 | 0.816 | 0.887 |
| seen_test_family_only | family_frequency | 0.000 | 0.000 | 0.236 | 0.764 |
| seen_test_family_only | visible_rule_based | 0.000 | 0.000 | 0.236 | 0.236 |
| unseen_family_test_request_only | family_frequency | 0.000 | 0.000 | 0.277 | 0.723 |
| unseen_family_test_request_only | visible_rule_based | 0.908 | 0.900 | 0.851 | 0.916 |
| unseen_family_test_no_constraints | family_frequency | 0.000 | 0.000 | 0.277 | 0.723 |
| unseen_family_test_no_constraints | visible_rule_based | 0.908 | 0.900 | 0.851 | 0.916 |
| unseen_family_test_family_only | family_frequency | 0.000 | 0.000 | 0.277 | 0.723 |
| unseen_family_test_family_only | visible_rule_based | 0.000 | 0.000 | 0.277 | 0.277 |

## Contract-Dependence Summary

### Bucket Distribution

| split | valid | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|---:|
| seen_test | 165 (23.6%) | 307 (43.9%) | 169 (24.1%) | 59 (8.4%) |
| unseen_family_test | 194 (27.7%) | 288 (41.1%) | 161 (23.0%) | 57 (8.1%) |

### Semantic Examples by Bucket

| split | request_obvious | contract_dependent | mixed |
|---|---:|---:|---:|
| seen_test | 35 (18.2%) | 120 (62.5%) | 37 (19.3%) |
| unseen_family_test | 66 (33.8%) | 101 (51.8%) | 28 (14.4%) |

### Visible Rule-Based on Contract-Dependent Subsets

| subset | semantic macro-F1 | semantic micro-F1 | critical semantic miss rate |
|---|---:|---:|---:|
| contract_dependent_seen_test | 0.619 | 0.873 | 0.225 |
| contract_dependent_unseen_test | 0.575 | 0.878 | 0.218 |
