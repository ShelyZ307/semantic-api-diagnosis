# Contract-Dependence Tightening Report

## Scope

This Version 1 tightening stage keeps the existing 5 seen and 5 unseen endpoint
families and the fixed 10-label taxonomy. It adds per-example visible contract
policies, uses plausible counterfactual values, removes `_mismatch` identifier
markers, and filters the training split for within-split near-duplicates.

The transparent visible-rule baseline now reads visible contract-local clauses
when they are present. Request-only and no-constraints views remove those
clauses, making the ablation meaningful.

## Visible Rule-Based Baseline: Before vs After

Each cell is `before -> after`.

| evaluation | micro-F1 | semantic macro-F1 | critical semantic miss rate | exact match |
|---|---:|---:|---:|---:|
| full seen_test | 0.888 -> 0.804 | 0.878 -> 0.866 | 0.193 -> 0.228 | 0.816 -> 0.657 |
| full unseen_family_test | 0.908 -> 0.923 | 0.900 -> 0.957 | 0.154 -> 0.076 | 0.851 -> 0.874 |
| seen_test request_only | 0.888 -> 0.607 | 0.878 -> 0.306 | 0.193 -> 0.436 | 0.816 -> 0.380 |
| unseen_family_test request_only | 0.908 -> 0.737 | 0.900 -> 0.625 | 0.154 -> 0.310 | 0.851 -> 0.549 |
| seen_test no_constraints | 0.888 -> 0.607 | 0.878 -> 0.306 | 0.193 -> 0.436 | 0.816 -> 0.380 |
| unseen_family_test no_constraints | 0.908 -> 0.737 | 0.900 -> 0.625 | 0.154 -> 0.310 | 0.851 -> 0.549 |
| contract_dependent_seen_test | 0.680 -> 0.588 | 0.619 -> 0.607 | 0.225 -> 0.283 | 0.462 -> 0.343 |
| contract_dependent_unseen_test | 0.701 -> 0.751 | 0.575 -> 0.623 | 0.218 -> 0.114 | 0.491 -> 0.559 |

## Full vs Shortcut Gap After Tightening

| split | full semantic macro-F1 | request_only semantic macro-F1 | decrease without contract constraints |
|---|---:|---:|---:|
| seen_test | 0.866 | 0.306 | 0.560 |
| unseen_family_test | 0.957 | 0.625 | 0.332 |

Before tightening, both decreases were `0.000`.

## Contract-Dependence Buckets

| split | bucket | before | after |
|---|---|---:|---:|
| seen_test | valid | 165 (23.6%) | 160 (22.9%) |
| seen_test | request_obvious | 307 (43.9%) | 309 (44.1%) |
| seen_test | contract_dependent | 169 (24.1%) | 175 (25.0%) |
| seen_test | mixed | 59 (8.4%) | 56 (8.0%) |
| unseen_family_test | valid | 194 (27.7%) | 192 (27.4%) |
| unseen_family_test | request_obvious | 288 (41.1%) | 291 (41.6%) |
| unseen_family_test | contract_dependent | 161 (23.0%) | 161 (23.0%) |
| unseen_family_test | mixed | 57 (8.1%) | 56 (8.0%) |

The bucket proportions remain similar because the label taxonomy and generation
mix remain focused. The main improvement is behavioral: contract-removal
ablations now cause substantial degradation instead of reproducing full-input
performance.

## Quality Gates

- Exact duplicate serialized inputs within each split: `0`
- Near-duplicate serialized-input pairs within each split at `>= 0.92`: `0`
- Exact duplicate serialized inputs across splits: `0`
- Exact duplicate contract-plus-request pairs across splits: `0`
- Near-duplicate serialized-input pairs across splits at `>= 0.92`: `0`
- Seen/unseen family leakage: none
- `_mismatch` marker occurrences in final splits: `0`
- Hidden validator exposure warnings: `0`
- All 10 labels remain represented in every main split

## Interpretation

The shortcut problem materially improved. A transparent baseline can still
solve request-obvious structural cases, which is expected, but it no longer
recovers full semantic performance after contract constraints are removed.

The remaining scientific risk is that the contract-local templates are still
synthetic and regular. Fine-tuned models must be evaluated on full, hard, and
shortcut views to show that they learned contract-conditioned diagnosis rather
than template matching.
