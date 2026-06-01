# Semantic API Request Diagnosis

This project investigates whether language models can diagnose semantic API request errors from partial or natural-language endpoint contracts.

The task is formulated as multi-label classification over API request error labels, with secondary predictions for request validity and severity. The central focus is semantic error diagnosis and generalization to held-out unseen endpoint families.

## Research Question

Can fine-tuned language models diagnose semantic API request errors from partial or natural-language endpoint contracts, and generalize to unseen endpoint families, better than rule-based validation and zero/few-shot LLM baselines?

## Task Formulation

Each example contains:

- an endpoint contract with natural-language constraints
- an API request
- a serialized text input for models
- target labels containing `validity`, `error_labels`, and `severity_bucket`

Primary task: multi-label classification over `error_labels`.

Secondary tasks:

- binary validity prediction
- severity bucket prediction

## Label Taxonomy

- `missing_required_field`
- `wrong_type`
- `invalid_value_range`
- `malformed_url`
- `wrong_http_method`
- `missing_authentication`
- `unexpected_or_malformed_body_structure`
- `semantic_cross_field_violation`
- `semantic_domain_constraint_violation`
- `semantic_state_violation`

`invalid_value_range` covers constrained values that are invalid for the contract. In Version 1 this includes numeric values outside an allowed range, enum/category values outside the allowed set, and unsupported values for constrained fields such as unknown appointment types, plans, priorities, roles, or payment methods.

## Current Status

- Controlled synthetic dataset generator implemented
- 5 seen endpoint families implemented
- 5 held-out unseen endpoint families implemented
- Final Version 1 dataset generated locally
- Leakage, duplicate, overlap, and shortcut checks passed
- Baseline evaluation framework implemented
- Contract-dependent hard subsets created
- Contract-dependence tightening with visible per-example policy counterfactuals implemented
- DistilBERT and RoBERTa fine-tuning implemented and run
- Validation-tuned encoder evaluation implemented for full, hard-subset, and shortcut views
- Positive-class-weighted RoBERTa tried once and rejected as an improvement because it overpredicts semantic errors
- Fixed 120-example Stage 8 LLM sample protocol implemented
- Provider-backed zero-shot/few-shot LLM results are not available unless `OPENAI_API_KEY` is set and the Stage 8 commands are run

## Main Findings

The honest Version 1 conclusion is narrower than the original hoped-for claim:

- RoBERTa learns contract-sensitive behavior and performs strongly on seen endpoint families.
- Removing contract text sharply hurts RoBERTa semantic performance, especially on seen-family examples.
- RoBERTa does not solve unseen-family semantic generalization.
- RoBERTa does not beat the visible rule-based baseline on unseen semantic diagnosis.
- The benchmark reveals that many examples are request-obvious; the most scientifically useful cases are the contract-dependent hard subsets and shortcut ablations.

The project contribution is therefore a controlled benchmark and analysis framework for contract-dependent semantic API diagnosis, not a claim that fine-tuned encoders fully solve held-out endpoint-family reasoning.

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

## Dataset Splits

The frozen Version 1 dataset is generated locally under `data/generated/`:

- `final_train.jsonl`: 3000 examples
- `final_validation.jsonl`: 500 examples
- `final_seen_test.jsonl`: 700 examples
- `final_unseen_family_test.jsonl`: 700 examples

Full generated JSONL files are intentionally ignored by git. Small representative samples live in `data/samples/`, and the full dataset can be regenerated with the commands below.

## Project Structure

```text
semantic-api-diagnosis/
├── semantic_api_diagnosis/          # Core package
├── scripts/                         # Generation, audit, leakage, and evaluation CLIs
├── tests/                           # Unit and integration tests
├── data/
│   ├── samples/                     # Small checked-in representative samples
│   ├── generated/                   # Local generated datasets and reports, ignored by git
│   ├── processed/                   # Local processed artifacts, ignored by git
│   └── raw/                         # Local raw artifacts, ignored by git
├── docs/
│   └── results/                     # Checked-in summary reports
├── configs/
├── README.md
├── pyproject.toml
└── .gitignore
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Core generation, baseline, and test code use the Python standard library plus `pytest`. Real encoder training/evaluation additionally requires the optional `train` dependencies.

## Running Tests

```bash
python3 -m pytest
```

## Generate the Final Dataset

Use the joint final-split generator instead of generating each split independently. It rejects cross-split near-duplicate serialized inputs before writing final files.

```bash
python3 scripts/generate_final_dataset.py \
  --train-count 3000 \
  --validation-count 500 \
  --seen-test-count 700 \
  --unseen-test-count 700 \
  --output-dir data/generated \
  --seed 101
```

## Quality Checks

```bash
python3 scripts/check_cross_split_leakage.py --splits \
  data/generated/final_train.jsonl \
  data/generated/final_validation.jsonl \
  data/generated/final_seen_test.jsonl \
  data/generated/final_unseen_family_test.jsonl

python3 scripts/check_seen_unseen_overlap.py \
  --seen data/generated/final_train.jsonl \
  --unseen data/generated/final_unseen_family_test.jsonl
```

## Baseline Evaluation

```bash
python3 scripts/evaluate_baselines.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --seen-test data/generated/final_seen_test.jsonl \
  --unseen-test data/generated/final_unseen_family_test.jsonl \
  --output data/generated/baseline_results.json
```

Reports include all-label micro/macro F1, per-label F1, semantic-only F1, exact-match label-set accuracy, validity accuracy, severity accuracy, and critical semantic-error miss rate.

## Contract-Dependent Hard Subsets

```bash
python3 scripts/analyze_contract_dependence.py \
  --train data/generated/final_train.jsonl \
  --seen-test data/generated/final_seen_test.jsonl \
  --unseen-test data/generated/final_unseen_family_test.jsonl \
  --output data/generated/contract_dependence_report.md
```

Model results are reported on both full test sets and contract-dependent hard subsets.

The Version 1 tightening comparison is recorded in
`docs/results/contract_dependence_tightening_report.md`. Request-only and
no-constraints views now show a substantial semantic-performance decrease
relative to full contract input.

## LLM Baseline Samples

LLM baseline infrastructure supports mock and dry-run modes. Tests do not require real API calls or credentials.

```bash
python3 scripts/run_llm_baseline.py \
  --mode zero_shot \
  --provider mock \
  --model mock-model \
  --input data/generated/final_seen_test.jsonl \
  --sample-size 10 \
  --output data/generated/llm_zero_shot_seen_mock_predictions.jsonl \
  --seed 201

python3 scripts/evaluate_llm_predictions.py \
  --predictions data/generated/llm_zero_shot_seen_mock_predictions.jsonl \
  --output data/generated/llm_zero_shot_seen_mock_results.json
```

Few-shot demonstrations must come from training examples only. Do not include validation, seen-test, unseen-family-test, or unseen-family examples as demonstrations.

## Stage 6: Fine-Tuning

Large LLMs are used only as zero/few-shot baselines. Fine-tuning is done on encoder models such as DistilBERT and RoBERTa.

The fine-tuning pipeline performs multi-label classification over the fixed error-label taxonomy. DistilBERT and RoBERTa were both trained and evaluated for Version 1. Full generated datasets and model checkpoints are not committed to git; regenerate the datasets locally before rerunning training.

Dry-run command:

```bash
python3 scripts/train_distilbert.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --dry-run true
```

Optional training dependencies can be installed with:

```bash
python3 -m pip install -e ".[train]"
```

### Stage 6B: Tiny Smoke Training

This is only a pipeline smoke test. It is not the final reported model. Outputs are intentionally ignored by git.

```bash
python3 scripts/train_distilbert.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --model-name distilbert-base-uncased \
  --output-dir outputs/distilbert_smoke \
  --epochs 1 \
  --batch-size 4 \
  --max-length 256 \
  --max-train-examples 32 \
  --max-validation-examples 16 \
  --dry-run false
```

### Stage 6C: Full Encoder Training And Evaluation

Reproduce the full DistilBERT experiment:

```bash
python3 scripts/train_distilbert.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --model-name distilbert-base-uncased \
  --output-dir outputs/distilbert_v1_seed42 \
  --epochs 3 \
  --batch-size 16 \
  --max-length 512 \
  --seed 42 \
  --save-final-model true \
  --dry-run false

python3 scripts/run_finetuned_evaluation.py \
  --model-dir outputs/distilbert_v1_seed42/model \
  --output-dir outputs/distilbert_v1_seed42/evaluation \
  --threshold-mode per_label \
  --max-length 512
```

Reproduce the RoBERTa experiment using the same generated train and validation splits:

```bash
python3 scripts/train_roberta.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --model-name roberta-base \
  --output-dir outputs/roberta_v1_seed42 \
  --epochs 3 \
  --batch-size 4 \
  --max-length 512 \
  --seed 42 \
  --save-final-model true \
  --dry-run false

python3 scripts/run_finetuned_evaluation.py \
  --model-dir outputs/roberta_v1_seed42/model \
  --output-dir outputs/roberta_v1_seed42/evaluation \
  --threshold-mode per_label \
  --max-length 512
```

The smaller RoBERTa batch size is the CPU-friendly setting used for the Version 1 local run. Each evaluator run tunes per-label thresholds on `final_validation.jsonl` only, saves them to `thresholds.json`, and applies them unchanged to full, contract-dependent, and shortcut-ablation test views.

### Stage 7: Controlled LLM And Weighted-Loss Comparison

Provider-backed OpenAI samples require `OPENAI_API_KEY`. The controlled runner evaluates zero-shot and few-shot prompts on sampled seen, unseen-family, and contract-dependent unseen examples. Raw responses, parsed predictions, failures, and metrics are saved under `data/generated/stage7_llm/`. No provider-backed LLM result is included unless this command is run with real credentials.

```bash
python3 scripts/run_stage7_llm_baselines.py \
  --provider openai \
  --model gpt-4o-mini \
  --sample-size 40
```

The single positive-class-weighted RoBERTa comparison was run and rejected as an improvement. It reduced critical semantic misses by predicting semantic labels too broadly, which severely hurt precision, exact match, and overall micro-F1. Reproduce it with:

```bash
python3 scripts/train_roberta.py \
  --train data/generated/final_train.jsonl \
  --validation data/generated/final_validation.jsonl \
  --model-name roberta-base \
  --output-dir outputs/roberta_weighted_v1_seed42 \
  --epochs 3 \
  --batch-size 4 \
  --max-length 512 \
  --seed 42 \
  --loss-mode pos_weighted \
  --save-final-model true \
  --dry-run false

python3 scripts/run_finetuned_evaluation.py \
  --model-dir outputs/roberta_weighted_v1_seed42/model \
  --output-dir outputs/roberta_weighted_v1_seed42/evaluation \
  --threshold-mode per_label \
  --max-length 512

python3 scripts/analyze_stage7_unseen_errors.py
python3 scripts/generate_stage7_report.py
```

### Stage 8: Fixed Sample LLM Baselines

Create the fixed cost-controlled sample and evaluate non-LLM baselines on exactly the same rows:

```bash
python3 scripts/create_stage8_llm_sample.py \
  --seed 808 \
  --per-source 40 \
  --output-dir data/generated/stage8_llm_sample

python3 scripts/evaluate_stage8_sample_non_llm.py \
  --output data/generated/stage8_llm_sample/non_llm_sample_metrics.json
```

Provider-backed LLM calls fail fast unless `OPENAI_API_KEY` is set. In the current local run, `OPENAI_API_KEY` was unavailable, so real zero-shot/few-shot LLM results are missing and mock outputs are excluded from scientific claims. When credentials are available, run:

```bash
python3 scripts/run_llm_baseline.py \
  --mode zero_shot \
  --provider openai \
  --model gpt-4o-mini \
  --input data/generated/stage8_llm_sample/combined_sample.jsonl \
  --sample-size 120 \
  --output data/generated/stage8_llm_sample/zero_shot_predictions.jsonl \
  --seed 808 \
  --timeout 60

python3 scripts/run_llm_baseline.py \
  --mode few_shot \
  --provider openai \
  --model gpt-4o-mini \
  --input data/generated/stage8_llm_sample/combined_sample.jsonl \
  --train data/generated/final_train.jsonl \
  --sample-size 120 \
  --output data/generated/stage8_llm_sample/few_shot_predictions.jsonl \
  --seed 808 \
  --timeout 60

python3 scripts/evaluate_stage8_llm_sample_predictions.py \
  --predictions data/generated/stage8_llm_sample/zero_shot_predictions.jsonl \
  --output data/generated/stage8_llm_sample/zero_shot_metrics.json

python3 scripts/evaluate_stage8_llm_sample_predictions.py \
  --predictions data/generated/stage8_llm_sample/few_shot_predictions.jsonl \
  --output data/generated/stage8_llm_sample/few_shot_metrics.json

python3 scripts/generate_stage8_llm_sample_report.py
```

## Documentation

See `docs/` for stage summaries and `docs/results/` for checked-in baseline, contract-dependence, fine-tuned model, weighted-loss, Stage 8 sample-protocol, and final summary reports.

## Remaining Work

Before final submission, the main optional missing experiment is to run provider-backed zero-shot and few-shot LLM baselines on the fixed Stage 8 sample if `OPENAI_API_KEY` is available. Do not present mock LLM results as scientific evidence.
