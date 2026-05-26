# Project Overview

Semantic API Request Diagnosis is a synthetic NLP benchmark for diagnosing API request errors from endpoint contracts and request payloads.

The primary task is multi-label prediction over a fixed API error taxonomy. The secondary outputs are request validity and severity bucket. The benchmark emphasizes semantic errors that require interpreting natural-language endpoint constraints, plus generalization from seen endpoint families to held-out unseen families.

Version 1 contains 5 seen endpoint families and 5 held-out unseen endpoint families. The final dataset is generated locally and intentionally not committed in full. Small representative samples are checked into `data/samples/`.

Key design goals:

- deterministic hidden validators for labels
- natural-language variation in endpoint contracts
- balanced endpoint-family sampling
- duplicate and leakage checks before modeling
- hard contract-dependent subsets for evaluation
- transparent rule-based and LLM baseline infrastructure
