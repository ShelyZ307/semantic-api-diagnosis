# Stage 7: LLM Baselines, Weighted RoBERTa, And Unseen Error Analysis

## Experiment Setup

- Dataset, endpoint families, label taxonomy, seed, and validation-only threshold policy are unchanged.
- Weighted RoBERTa is exactly one controlled improvement attempt using training-split `negatives / positives` weights in multi-label BCE.
- Real LLM rows are reported only when provider-backed artifacts exist. Mock outputs are excluded.

## Main Comparison

| view | model | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate | semantic cross-field F1 | semantic domain F1 | semantic state F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seen_test | visible rule baseline | 0.804 | 0.866 | 0.657 | 0.228 | 0.757 | 0.840 | 1.000 |
| seen_test | DistilBERT | 0.662 | 0.281 | 0.374 | 0.564 | 0.270 | 0.326 | 0.246 |
| seen_test | original RoBERTa | 0.949 | 0.807 | 0.903 | 0.272 | 0.672 | 0.750 | 1.000 |
| seen_test | weighted-loss RoBERTa | 0.239 | 0.196 | 0.064 | 0.000 | 0.191 | 0.206 | 0.192 |
| seen_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| seen_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| unseen_family_test | visible rule baseline | 0.923 | 0.957 | 0.874 | 0.076 | 1.000 | 0.872 | 1.000 |
| unseen_family_test | DistilBERT | 0.587 | 0.179 | 0.403 | 0.782 | 0.317 | 0.047 | 0.173 |
| unseen_family_test | original RoBERTa | 0.721 | 0.352 | 0.606 | 0.558 | 0.439 | 0.051 | 0.566 |
| unseen_family_test | weighted-loss RoBERTa | 0.226 | 0.192 | 0.053 | 0.000 | 0.181 | 0.199 | 0.196 |
| unseen_family_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| unseen_family_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_unseen_test | visible rule baseline | 0.751 | 0.623 | 0.559 | 0.114 | 0.000 | 0.870 | 1.000 |
| contract_dependent_unseen_test | DistilBERT | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| contract_dependent_unseen_test | original RoBERTa | 0.420 | 0.263 | 0.217 | 0.524 | 0.000 | 0.069 | 0.719 |
| contract_dependent_unseen_test | weighted-loss RoBERTa | 0.196 | 0.328 | 0.000 | 0.000 | 0.000 | 0.488 | 0.495 |
| contract_dependent_unseen_test | zero-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contract_dependent_unseen_test | few-shot LLM | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Weighted-Loss Delta

| view | metric | original RoBERTa | weighted RoBERTa | delta |
|---|---|---:|---:|---:|
| seen_test | micro-F1 | 0.949 | 0.239 | -0.709 |
| seen_test | micro precision | 0.980 | 0.138 | -0.842 |
| seen_test | semantic macro-F1 | 0.807 | 0.196 | -0.611 |
| seen_test | semantic domain F1 | 0.750 | 0.206 | -0.544 |
| seen_test | exact match | 0.903 | 0.064 | -0.839 |
| seen_test | critical miss rate | 0.272 | 0.000 | -0.272 |
| unseen_family_test | micro-F1 | 0.721 | 0.226 | -0.495 |
| unseen_family_test | micro precision | 0.800 | 0.130 | -0.670 |
| unseen_family_test | semantic macro-F1 | 0.352 | 0.192 | -0.160 |
| unseen_family_test | semantic domain F1 | 0.051 | 0.199 | +0.148 |
| unseen_family_test | exact match | 0.606 | 0.053 | -0.553 |
| unseen_family_test | critical miss rate | 0.558 | 0.000 | -0.558 |
| contract_dependent_seen_test | micro-F1 | 0.927 | 0.216 | -0.711 |
| contract_dependent_seen_test | micro precision | 1.000 | 0.126 | -0.874 |
| contract_dependent_seen_test | semantic macro-F1 | 0.839 | 0.380 | -0.459 |
| contract_dependent_seen_test | semantic domain F1 | 0.783 | 0.485 | -0.298 |
| contract_dependent_seen_test | exact match | 0.840 | 0.000 | -0.840 |
| contract_dependent_seen_test | critical miss rate | 0.220 | 0.000 | -0.220 |
| contract_dependent_unseen_test | micro-F1 | 0.420 | 0.196 | -0.224 |
| contract_dependent_unseen_test | micro precision | 0.634 | 0.114 | -0.520 |
| contract_dependent_unseen_test | semantic macro-F1 | 0.263 | 0.328 | +0.065 |
| contract_dependent_unseen_test | semantic domain F1 | 0.069 | 0.488 | +0.419 |
| contract_dependent_unseen_test | exact match | 0.217 | 0.000 | -0.217 |
| contract_dependent_unseen_test | critical miss rate | 0.524 | 0.000 | -0.524 |

## Shortcut Ablation

| model | split | full semantic macro-F1 | request-only | no constraints | request-only delta | no-constraints delta |
|---|---|---:|---:|---:|---:|---:|
| original RoBERTa | seen_test | 0.807 | 0.200 | 0.184 | -0.607 | -0.623 |
| original RoBERTa | unseen_family_test | 0.352 | 0.123 | 0.129 | -0.229 | -0.223 |
| weighted-loss RoBERTa | seen_test | 0.196 | 0.186 | 0.170 | -0.010 | -0.026 |
| weighted-loss RoBERTa | unseen_family_test | 0.192 | 0.178 | 0.187 | -0.014 | -0.005 |

## Real LLM Baseline Status

- Stage 7 itself has no full-test provider-backed LLM artifacts. A later Stage 8 `gpt-4o-mini` fixed-sample run completed on 120 examples and is reported separately in `docs/results/stage_8_llm_sample_baselines.md`. Mock results remain excluded from scientific comparison.

## Interpretation

- Original RoBERTa beats the visible rule baseline on seen full-input micro-F1 (`0.949` vs `0.804`) but not on unseen semantic diagnosis (`0.352` vs `0.957` semantic macro-F1).
- Weighted loss changes unseen semantic macro-F1 by `-0.160`, unseen semantic-domain F1 by `+0.148`, and critical semantic miss rate by `-0.558`. It is rejected: the zero miss rate comes from broad overprediction, not better diagnosis.
- Original RoBERTa is contract-sensitive: its semantic macro-F1 falls sharply on request-only and no-constraints views. Weighted RoBERTa loses that useful ablation pattern because its predictions are over-broad.
- Real zero-shot and few-shot LLM comparisons are available only as the later Stage 8 fixed-sample baseline. They do not replace the Stage 7 full-test encoder comparison.
- The defensible Stage 7 conclusion is that fine-tuned RoBERTa learns contract-sensitive behavior on seen families, while unseen-family transfer remains weak and the visible rule baseline remains stronger on held-out domains.

See `docs/results/stage_7_unseen_error_analysis.md` for grouped failure analysis and concrete examples.
