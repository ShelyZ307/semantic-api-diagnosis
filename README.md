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
- Zero-shot/few-shot LLM baseline infrastructure implemented with mock/dry-run support
- Ready for fine-tuned model training

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

The project currently uses the Python standard library plus `pytest` for tests.

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

Future model results should be reported on both full test sets and contract-dependent hard subsets.

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

## Stage 6: Fine-tuning

Large LLMs are used only as zero/few-shot baselines. Fine-tuning is done on encoder models such as DistilBERT, RoBERTa, or DeBERTa.

The first fine-tuning scaffold is DistilBERT multi-label classification over the fixed error-label taxonomy. Full generated datasets are not committed to git; regenerate them locally before running real training.

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

## Documentation

See `docs/` for stage summaries and `docs/results/` for checked-in baseline and contract-dependence reports.

## Next Stage

Fine-tuned model training with DistilBERT, RoBERTa, and DeBERTa. Transformer training scripts are intentionally placeholder-only until baseline and leakage gates are finalized.
