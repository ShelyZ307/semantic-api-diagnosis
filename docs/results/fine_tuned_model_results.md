# Fine-Tuned Encoder Results

## Experiment Setup

Version 1 uses the tightened contract-dependence dataset without adding endpoint families or changing the label taxonomy.

| split | examples |
|---|---:|
| train | 3000 |
| validation | 500 |
| seen_test | 700 |
| unseen_family_test | 700 |
| contract_dependent_seen_test | 175 |
| contract_dependent_unseen_test | 161 |

- Labels: 10 multi-label diagnosis targets: invalid_value_range, malformed_url, missing_authentication, missing_required_field, semantic_cross_field_violation, semantic_domain_constraint_violation, semantic_state_violation, unexpected_or_malformed_body_structure, wrong_http_method, wrong_type.
- Encoders: `distilbert-base-uncased` and `roberta-base` when a completed checkpoint is available.
- Training: three epochs, fixed seed `42`, maximum sequence length `512`, validation on `final_validation.jsonl` only. DistilBERT used batch size `16`; RoBERTa used the CPU-feasible batch size `4`.
- Thresholding: per-label thresholds selected on validation macro-F1 and reused unchanged for every test and shortcut view.
- Auxiliary validity and severity metrics are derived from predicted error labels; there are no separate auxiliary heads.
- LLM baseline code exists, but no real provider-backed zero-shot or few-shot result is available. Mock output is excluded from scientific comparisons.

## Main Results

| view | model | micro-F1 | macro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |
|---|---|---:|---:|---:|---:|---:|
| seen_test | visible rule-based | 0.804 | 0.802 | 0.866 | 0.657 | 0.228 |
| seen_test | DistilBERT | 0.662 | 0.719 | 0.281 | 0.374 | 0.564 |
| seen_test | RoBERTa | 0.949 | 0.942 | 0.807 | 0.903 | 0.272 |
| seen_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| seen_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| unseen_family_test | visible rule-based | 0.923 | 0.887 | 0.957 | 0.874 | 0.076 |
| unseen_family_test | DistilBERT | 0.587 | 0.563 | 0.179 | 0.403 | 0.782 |
| unseen_family_test | RoBERTa | 0.721 | 0.689 | 0.352 | 0.606 | 0.558 |
| unseen_family_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| unseen_family_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_seen_test | visible rule-based | 0.588 | 0.213 | 0.607 | 0.343 | 0.283 |
| contract_dependent_seen_test | DistilBERT | 0.521 | 0.254 | 0.181 | 0.057 | 0.717 |
| contract_dependent_seen_test | RoBERTa | 0.927 | 0.452 | 0.839 | 0.840 | 0.220 |
| contract_dependent_seen_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_seen_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_unseen_test | visible rule-based | 0.751 | 0.287 | 0.623 | 0.559 | 0.114 |
| contract_dependent_unseen_test | DistilBERT | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| contract_dependent_unseen_test | RoBERTa | 0.420 | 0.119 | 0.263 | 0.217 | 0.524 |
| contract_dependent_unseen_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_unseen_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a |

Real zero-shot and few-shot LLM values are `n/a` because the current scaffold has no provider-backed experiment artifact.

## Threshold Comparison

| model | view | thresholding | micro-F1 | macro-F1 | semantic macro-F1 | exact match |
|---|---|---|---:|---:|---:|---:|
| DistilBERT | seen_test | default_0_5 | 0.727 | 0.582 | 0.000 | 0.587 |
| DistilBERT | seen_test | tuned | 0.662 | 0.719 | 0.281 | 0.374 |
| DistilBERT | unseen_family_test | default_0_5 | 0.649 | 0.500 | 0.000 | 0.559 |
| DistilBERT | unseen_family_test | tuned | 0.587 | 0.563 | 0.179 | 0.403 |
| RoBERTa | seen_test | default_0_5 | 0.918 | 0.868 | 0.559 | 0.851 |
| RoBERTa | seen_test | tuned | 0.949 | 0.942 | 0.807 | 0.903 |
| RoBERTa | unseen_family_test | default_0_5 | 0.711 | 0.640 | 0.189 | 0.599 |
| RoBERTa | unseen_family_test | tuned | 0.721 | 0.689 | 0.352 | 0.606 |

## Shortcut Ablation

| model | test split | full semantic macro-F1 | request-only | no constraints | request-only delta | no-constraints delta |
|---|---|---:|---:|---:|---:|---:|
| DistilBERT | seen_test | 0.281 | 0.229 | 0.252 | -0.051 | -0.028 |
| DistilBERT | unseen_family_test | 0.179 | 0.255 | 0.247 | +0.076 | +0.068 |
| RoBERTa | seen_test | 0.807 | 0.200 | 0.184 | -0.607 | -0.623 |
| RoBERTa | unseen_family_test | 0.352 | 0.123 | 0.129 | -0.229 | -0.223 |

## Error Analysis

### DistilBERT

Weakest unseen-family labels: `missing_required_field` F1=0.000, `semantic_domain_constraint_violation` F1=0.047, `semantic_state_violation` F1=0.173.

Cross-field/domain substitution cases on unseen families: 0.

Sample unseen-family semantic false negatives:
- `ex_000002` (`healthcare_appointments`): missed `semantic_cross_field_violation`; predicted `no errors`.
- `ex_000003` (`course_registration`): missed `semantic_domain_constraint_violation`; predicted `no errors`.
- `ex_000004` (`ticket_support_workflow`): missed `semantic_state_violation`; predicted `no errors`.

### RoBERTa

Weakest unseen-family labels: `semantic_domain_constraint_violation` F1=0.051, `missing_required_field` F1=0.412, `semantic_cross_field_violation` F1=0.439.

Cross-field/domain substitution cases on unseen families: 2.

Sample unseen-family semantic false negatives:
- `ex_000002` (`healthcare_appointments`): missed `semantic_cross_field_violation`; predicted `no errors`.
- `ex_000003` (`course_registration`): missed `semantic_domain_constraint_violation`; predicted `missing_authentication`.
- `ex_000016` (`healthcare_appointments`): missed `semantic_domain_constraint_violation`; predicted `no errors`.

## Auxiliary Metrics

| model | view | validity accuracy | validity F1 | severity accuracy |
|---|---|---:|---:|---:|
| DistilBERT | seen_test | 0.801 | 0.853 | 0.664 |
| DistilBERT | unseen_family_test | 0.706 | 0.749 | 0.550 |
| RoBERTa | seen_test | 0.950 | 0.967 | 0.927 |
| RoBERTa | unseen_family_test | 0.780 | 0.836 | 0.749 |

## Interpretation

- DistilBERT: semantic macro-F1 versus the visible rule baseline changes by -0.585 on seen families and -0.779 on unseen families. Removing contract text changes seen semantic macro-F1 by -0.051.
- RoBERTa: semantic macro-F1 versus the visible rule baseline changes by -0.058 on seen families and -0.605 on unseen families. Removing contract text changes seen semantic macro-F1 by -0.607.
- Do fine-tuned models beat the visible rule baseline? Not consistently. RoBERTa beats it on all-label seen-family metrics and on the contract-dependent seen subset, but not on semantic seen-family macro-F1 and not on unseen families.
- Do fine-tuned models improve on contract-dependent examples? RoBERTa does on seen families. The unseen-family hard subset remains weak.
- Do models rely on contract text? RoBERTa clearly does: removing constraints produces a large semantic macro-F1 drop on seen and unseen tests. DistilBERT shows only a small seen drop and an inconsistent unseen ablation.
- Does Version 1 support the full research claim? No. The results support a narrower claim that a fine-tuned RoBERTa encoder learns contract-dependent diagnosis on seen families and partially transfers to unseen families, while the transparent rule baseline remains substantially stronger on unseen-family semantic diagnosis.

## Reproducibility

Artifacts are saved under `outputs/distilbert_v1_seed42/` and `outputs/roberta_v1_seed42/` when each model run completes. Each evaluation directory contains `thresholds.json`, per-view metric JSON files, per-example predictions, `summary.json`, and an error-analysis sample.
