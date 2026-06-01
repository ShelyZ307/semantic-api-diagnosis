# Stage 8: Sampled Real LLM Baselines

## Sample Description

- Total sample size: `120`.
- Sampling method: greedy stratified coverage of validity, multi-label cases, semantic labels, and structural labels, followed by seeded random fill.
- Seed: `808`.
- Source groups: `40` seen, `40` unseen-family, `40` contract-dependent unseen.
- Limitation: this is a cost-controlled sample, so it complements but does not replace the full-test encoder and rule-baseline evaluation.

### seen sample

- Validity: `{'invalid': 34, 'valid': 6}`.
- Labels: `{'invalid_value_range': 6, 'malformed_url': 4, 'missing_authentication': 5, 'missing_required_field': 5, 'semantic_cross_field_violation': 7, 'semantic_domain_constraint_violation': 3, 'semantic_state_violation': 2, 'unexpected_or_malformed_body_structure': 3, 'wrong_http_method': 4, 'wrong_type': 6}`.
- Semantic labels: `{'semantic_cross_field_violation': 7, 'semantic_domain_constraint_violation': 3, 'semantic_state_violation': 2}`.
- Endpoint families: `{'booking_reservation': 7, 'inventory_product_search': 13, 'payments_invoices': 6, 'refunds_orders': 6, 'user_permissions': 8}`.
- Single vs multi-error: `{'multi_error': 11, 'single_error': 23, 'valid': 6}`.

### unseen sample

- Validity: `{'invalid': 30, 'valid': 10}`.
- Labels: `{'invalid_value_range': 3, 'malformed_url': 4, 'missing_authentication': 4, 'missing_required_field': 4, 'semantic_cross_field_violation': 3, 'semantic_domain_constraint_violation': 3, 'semantic_state_violation': 4, 'unexpected_or_malformed_body_structure': 1, 'wrong_http_method': 7, 'wrong_type': 6}`.
- Semantic labels: `{'semantic_cross_field_violation': 3, 'semantic_domain_constraint_violation': 3, 'semantic_state_violation': 4}`.
- Endpoint families: `{'course_registration': 5, 'healthcare_appointments': 6, 'shipping_returns': 11, 'subscription_plan_changes': 8, 'ticket_support_workflow': 10}`.
- Single vs multi-error: `{'multi_error': 9, 'single_error': 21, 'valid': 10}`.

### contract-dependent unseen sample

- Validity: `{'invalid': 40}`.
- Labels: `{'invalid_value_range': 5, 'missing_required_field': 12, 'semantic_domain_constraint_violation': 15, 'semantic_state_violation': 15}`.
- Semantic labels: `{'semantic_domain_constraint_violation': 15, 'semantic_state_violation': 15}`.
- Endpoint families: `{'course_registration': 8, 'healthcare_appointments': 6, 'shipping_returns': 11, 'subscription_plan_changes': 8, 'ticket_support_workflow': 7}`.
- Single vs multi-error: `{'multi_error': 7, 'single_error': 33}`.
- Note: this source subset contains no `semantic_cross_field_violation` examples, so that semantic label cannot be represented here.

## Main Sampled Comparison

| group | model | n | micro-F1 | semantic macro-F1 | exact match | critical semantic miss rate |
|---|---|---:|---:|---:|---:|---:|
| seen sample | visible rule baseline | 40 | 0.864 | 0.974 | 0.775 | 0.083 |
| seen sample | original RoBERTa | 40 | 0.955 | 0.774 | 0.900 | 0.250 |
| seen sample | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| seen sample | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| unseen sample | visible rule baseline | 40 | 0.946 | 1.000 | 0.900 | 0.000 |
| unseen sample | original RoBERTa | 40 | 0.789 | 0.409 | 0.675 | 0.500 |
| unseen sample | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| unseen sample | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract-dependent unseen sample | visible rule baseline | 40 | 0.795 | 0.615 | 0.650 | 0.133 |
| contract-dependent unseen sample | original RoBERTa | 40 | 0.457 | 0.327 | 0.250 | 0.567 |
| contract-dependent unseen sample | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| contract-dependent unseen sample | few-shot LLM | n/a | n/a | n/a | n/a | n/a |
| combined sample | visible rule baseline | 120 | 0.867 | 0.947 | 0.775 | 0.096 |
| combined sample | original RoBERTa | 120 | 0.751 | 0.591 | 0.608 | 0.481 |
| combined sample | zero-shot LLM | n/a | n/a | n/a | n/a | n/a |
| combined sample | few-shot LLM | n/a | n/a | n/a | n/a | n/a |

## Per-Semantic-Label F1

| group | model | cross-field F1 | domain F1 | state F1 |
|---|---|---:|---:|---:|
| seen sample | visible rule baseline | 0.923 | 1.000 | 1.000 |
| seen sample | original RoBERTa | 0.923 | 0.400 | 1.000 |
| seen sample | zero-shot LLM | n/a | n/a | n/a |
| seen sample | few-shot LLM | n/a | n/a | n/a |
| unseen sample | visible rule baseline | 1.000 | 1.000 | 1.000 |
| unseen sample | original RoBERTa | 0.500 | 0.000 | 0.727 |
| unseen sample | zero-shot LLM | n/a | n/a | n/a |
| unseen sample | few-shot LLM | n/a | n/a | n/a |
| contract-dependent unseen sample | visible rule baseline | 0.000 | 0.846 | 1.000 |
| contract-dependent unseen sample | original RoBERTa | 0.000 | 0.222 | 0.759 |
| contract-dependent unseen sample | zero-shot LLM | n/a | n/a | n/a |
| contract-dependent unseen sample | few-shot LLM | n/a | n/a | n/a |
| combined sample | visible rule baseline | 0.947 | 0.895 | 1.000 |
| combined sample | original RoBERTa | 0.778 | 0.222 | 0.773 |
| combined sample | zero-shot LLM | n/a | n/a | n/a |
| combined sample | few-shot LLM | n/a | n/a | n/a |

## LLM Parse And Usage Quality

| model | group | parse failure rate | invalid JSON rate | transport error rate | usage totals |
|---|---|---:|---:|---:|---|
| zero-shot LLM | all | n/a | n/a | n/a | no provider-backed artifact |
| few-shot LLM | all | n/a | n/a | n/a | no provider-backed artifact |

## Interpretation

- On this fixed sample, original RoBERTa unseen semantic macro-F1 is `0.409` and the visible rule baseline is `1.000`.
- Real zero-shot/few-shot conclusions are still blocked because provider-backed LLM artifacts are missing. Do not treat mock or missing rows as scientific evidence.
- Recommended framing: RoBERTa learns contract-sensitive behavior and performs strongly in-domain, but unseen-family transfer remains weak. The project contribution is a controlled benchmark and analysis of contract-dependent semantic API diagnosis, not a claim that encoders solve unseen semantic generalization.
