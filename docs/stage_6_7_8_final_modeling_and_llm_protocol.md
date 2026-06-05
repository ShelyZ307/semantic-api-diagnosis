# Stage 6-8 Final Modeling And LLM Protocol

This document summarizes the final modeling stages for Version 1 of Semantic API Request Diagnosis from Partial or Natural-Language Endpoint Contracts.

## Stage 6: Real Fine-Tuning And Evaluation

Stage 6 moved the project from scaffolding to real encoder experiments.

Completed:

- Implemented multi-label encoder fine-tuning over the fixed 10-label taxonomy.
- Trained `distilbert-base-uncased` on `data/generated/final_train.jsonl`.
- Trained `roberta-base` on the same training split.
- Tuned decision thresholds on `data/generated/final_validation.jsonl` only.
- Evaluated on full seen, full unseen-family, contract-dependent seen, contract-dependent unseen, request-only, and no-constraints views.
- Saved local checkpoints and evaluation artifacts under `outputs/`.
- Wrote the checked-in summary report at `docs/results/fine_tuned_model_results.md`.

Main finding:

- RoBERTa performs strongly in-domain and clearly uses contract text.
- RoBERTa does not beat the visible rule-based baseline on unseen-family semantic diagnosis.
- DistilBERT is substantially weaker than RoBERTa and does not provide a convincing final model.

Key RoBERTa results:

| view | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |
|---|---:|---:|---:|---:|
| seen_test | 0.949 | 0.807 | 0.903 | 0.272 |
| unseen_family_test | 0.721 | 0.352 | 0.606 | 0.558 |
| contract_dependent_seen_test | 0.927 | 0.839 | 0.840 | 0.220 |
| contract_dependent_unseen_test | 0.420 | 0.263 | 0.217 | 0.524 |

## Stage 7: Weighted RoBERTa And Unseen Error Analysis

Stage 7 tested exactly one motivated improvement attempt: positive-class-weighted BCE for RoBERTa.

Completed:

- Added positive-class-weighted training support.
- Trained one weighted-loss RoBERTa run with the same dataset, seed, maximum length, and validation-threshold protocol.
- Evaluated weighted RoBERTa on the same full, hard-subset, and shortcut views.
- Generated unseen-family error analysis at `docs/results/stage_7_unseen_error_analysis.md`.
- Generated Stage 7 comparison report at `docs/results/stage_7_llm_and_weighted_roberta_results.md`.

Main finding:

- Weighted RoBERTa is rejected as an improvement.
- It eliminates critical semantic misses by predicting semantic labels too broadly.
- This causes a large precision and exact-match collapse.

Key weighted-loss comparison:

| view | original RoBERTa semantic macro-F1 | weighted RoBERTa semantic macro-F1 | original exact | weighted exact |
|---|---:|---:|---:|---:|
| seen_test | 0.807 | 0.196 | 0.903 | 0.064 |
| unseen_family_test | 0.352 | 0.192 | 0.606 | 0.053 |
| contract_dependent_unseen_test | 0.263 | 0.328 | 0.217 | 0.000 |

The contract-dependent unseen semantic macro-F1 increase is not a usable improvement because exact match drops to zero and predictions are over-broad.

## Stage 8: Fixed Sample LLM Baseline Protocol

Stage 8 prepares a cost-controlled real LLM comparison without using mock outputs as evidence.

Completed:

- Created a fixed 120-example sample with seed `808`.
- Sample includes:
  - 40 examples from `final_seen_test.jsonl`
  - 40 examples from `final_unseen_family_test.jsonl`
  - 40 examples from `contract_dependent_unseen_test.jsonl`
- Evaluated visible rule baseline and original RoBERTa on exactly the same sampled rows.
- Implemented provider-backed OpenAI command path with strict JSON output and fail-fast credential checks.
- Added retry/resume support after the provider-backed run encountered rate limits.
- Generated the Stage 8 report at `docs/results/stage_8_llm_sample_baselines.md`.

Current LLM status:

- Provider-backed `gpt-4o-mini` calls were attempted on all 120 fixed-sample examples.
- A later quota-safe resume completed the fixed sample with no transport or parse failures.
- Successful responses: zero-shot 120 / 120; few-shot 120 / 120.
- Results are documented as a clean fixed-sample LLM baseline, not as a full-test LLM benchmark.
- Mock LLM outputs are not used as scientific evidence.

Same-sample non-LLM results:

| group | visible rule semantic macro-F1 | RoBERTa semantic macro-F1 |
|---|---:|---:|
| seen sample | 0.974 | 0.774 |
| unseen sample | 1.000 | 0.409 |
| contract-dependent unseen sample | 0.615 | 0.327 |
| combined sample | 0.947 | 0.591 |

## Final Interpretation

The strongest supported conclusion is not that RoBERTa beats all baselines.

The supported conclusion is:

> The benchmark reveals that many examples are request-obvious and that visible rule-based validation is strong. RoBERTa learns contract-sensitive behavior and performs strongly in-domain, but unseen-family semantic generalization remains difficult. The most scientifically valuable part of Version 1 is the controlled dataset, contract-dependent subsets, shortcut views, and error analysis.

## Remaining Missing Item

There is no required remaining model experiment for final submission. The provider-backed zero-shot/few-shot LLM baseline is now complete only for the fixed Stage 8 sample; it should remain clearly separated from the full-test local encoder comparison.
